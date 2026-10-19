//! Replays staged [`Program`]s through chosen value semantics.
//!
//! Interpretation is the [`Program`] walk shared by eager execution, restaging, and transforms. It validates the
//! program boundary, seeds a flat environment with input values, lifts live constants, visits [`Instruction`]s in
//! order, and routes every operation through [`Context::bind`]. The supplied context determines what each bind means:
//! an [`EagerContext`] executes an eager rule, a [`StagingContext`](crate::StagingContext) records an equivalent
//! instruction, and a transform context applies its operation-owned rewrite. After replay, interpretation validates
//! output refinements and rebuilds the declared output structure. Refer to [`InterpretationDriver`] for a rendered
//! diagram of this pipeline and its nested-region path.
//!
//! # Choosing an Entry Point
//!
//!   - [`Program::interpret`] is the ordinary eager entry point. It constructs the program's [`EagerContext`], checks
//!     the structured input contract, executes every instruction, and reconstructs the structured result.
//!   - [`Program::interpret_in_context`] replays through an explicit compatible [`Context`]. Use it to stage a program
//!     into an active trace, apply a transform, or use backend-owned value semantics while preserving the same program
//!     traversal and boundary validation.
//!   - [`Program::interpret_with`] is the low-level flat replay fold. Infrastructure supplies constant-lifting and
//!     instruction-dispatch closures directly; it is responsible only for atom availability, instruction order,
//!     last-use value transfer, output counts, and flat output gathering.
//!
//! [`RegionRef`] provides corresponding context-driven and low-level functions that replay a borrowed sealed region
//! directly from its source arena. Context-driven [`Program`] and [`RegionRef`] calls share one flat replay
//! implementation. [`Program`] adds parameter-structure validation and reconstruction while retaining program-specific
//! diagnostics.
//!
//! # Replay Environment and Boundaries
//!
//! [`Program::interpret_in_context`] first checks the input [`Parameterized`] structure and the complete input type
//! signature. Refinements are established across the whole signature so repeated dynamic type identities cannot receive
//! contradictory concrete bindings. Structured inputs are then flattened into an atom-indexed environment. Only live
//! constants are lifted through [`Context::lift`], and every instruction reads its inputs and writes its results in
//! that environment. After validating the original boundary, staged replay carries permitted input definition
//! [`TypeIdentityRenaming`]s into operation payloads and nested regions. Fresh instruction definitions extend that
//! [`TypeIdentityRenaming`]. Staging also freshens local constant definitions so separate retained replays can share
//! one destination safely.
//!
//! Values are moved from the environment on their final use and cloned only when another consumer remains. Once all
//! instructions have run, output types are validated against the input refinement environment and the program's closed
//! type-identity signature. Flat outputs are finally reparameterized into the program's declared output structure.
//! Malformed atom identifiers, boundary mismatches, refinement conflicts, bind failures, and wrong operation-result
//! counts remain structured [`ProgramError`]s.
//!
//! # Context Semantics and Operation Rules
//!
//! Interpretation owns traversal and value liveness. [`Context`] owns the semantics of lifting and binding.
//! [`InterpretableOperation`] owns eager primitive or higher-order execution for one operation payload. This division
//! lets the same replay evaluate through an [`EagerContext`], append instructions through a staging context, or invoke
//! batching, differentiation, and partial-evaluation rules through transform contexts.
//!
//! Eager rules should request only the context capabilities they consume. Rules that consume inputs often need only
//! those input values, while nullary constructors and captured constants may require focused capabilities such as zero
//! construction or constant materialization. The generic context parameter on [`InterpretableOperation`] is therefore
//! deliberately bounded by [`Domain`] rather than by [`Context`]. The trait's type-compatibility caveat is documented
//! on the trait itself.
//!
//! # Nested Regions
//!
//! Every [`Context::bind`] receives one instruction-scoped [`BindingRegionDriver`]. Eager binding adapts that driver
//! into an [`InterpretationDriver`] for the operation's eager rule. Region-free applications use an empty driver.
//! Higher-order eager rules select attached regions by index and recursively replay them through the same active
//! context. Borrowed regions remain in the source arena, and shared replay mappings preserve repeated roots and
//! descendants when another active context stages them into a destination arena. No standalone program is materialized
//! for recursive eager interpretation. Rules that execute nested work instruction by instruction instead of replaying
//! a region verbatim (e.g., the `shard_map` rule, which emulates one replay per device) bind each operation through
//! [`InterpretationDriver::bind`], which applies it through the active context on their behalf.
//!
//! # Composite and Projected Operations
//!
//! [`InterpretableOperation`] describes eager semantics in an operation's native value universe.
//! [`MemberInterpretableOperation`] handles a payload whose enclosing instruction consumes or produces several member
//! kinds. [`interpret_projected_operation`] adapts a region-free native member rule to a composite domain by projecting
//! inputs, executing once through the member's eager context, and lifting results back into the composite value family.
//! Region-carrying and genuinely mixed operations remain composite-native because their region contracts may span
//! several member kinds.
//!
//! # Extending Interpretation
//!
//! Add eager semantics by implementing [`InterpretableOperation`] on operation payloads and dispatching
//! wrapper operation families to those implementations. Higher-order rules recurse through their supplied
//! [`InterpretationDriver`]. A new replay meaning normally requires no second interpreter: define a [`Context`]
//! whose [`Context::bind`] implements that meaning and call [`Program::interpret_in_context`].

use std::cell::RefCell;
use std::fmt::Debug;

use crate::contexts::{Context, Domain, EagerContext};
use crate::macros::check_count;
use crate::operations::GuardOperation;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{
    Atom, AtomId, BindingRegionDriver, EmptyRegionDriver, Instruction, Operation, OperationProvider, Program,
    ProgramError, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver, Type, TypeError,
    TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueProjection,
};

/// Provides instruction-scoped access to the attached [`Region`](crate::Region)s of one interpreted [`Operation`]
/// application. During [`Program`] replay, the application is exactly one [`Instruction`]. Direct rule invocation
/// supplies an equivalent [`RegionDriver`] for that call. Region-free applications use an empty driver, while
/// higher-order rules select borrowed attached regions and re-enter the active interpreter without materializing
/// standalone programs.
///
/// # Replay Pipeline
///
/// ```mermaid
/// flowchart TD
///   inputs["Structured Inputs"] --> environment["Flat Atom Environment"]
///   constants["Live Program Constants"] -->|"lift through active context"| environment
///   environment --> instruction["Next Instruction in Program Order"]
///   regions["Instruction Attached Regions"] --> binding_driver["Application-Scoped Binding Region Driver"]
///   instruction --> bind["Context Bind"]
///   binding_driver --> bind
///   bind --> eager["Eager Context"]
///   bind --> staging["Staging Context"]
///   bind --> transform["Transform Context"]
///   eager --> driver["Adapt Attached Regions as Interpretation Driver"]
///   driver --> rule["InterpretableOperation Rule"]
///   staging --> staged["Import Regions and Append Equivalent Instruction"]
///   transform --> transformed["Apply Transform Rule"]
///   rule -->|"higher-order rule requests a region"| recurse["Replay Borrowed Region Through Same Eager Context"]
///   rule --> results["Instruction Results"]
///   staged --> results
///   transformed --> results
///   recurse --> results
///   results --> environment
///   environment -->|"after final instruction"| outputs["Validate Flat Outputs and Reconstruct Declared Structure"]
/// ```
///
/// The driver is application-scoped rather than stored in the operation payload, so the same operation can receive
/// owned, borrowed, replayed, or shared attached computations through one ordered region contract.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait InterpretationDriver<C: Domain>: RegionDriver<C::Value, C::Operation> {
    /// Interprets the [`Region`](crate::Region) at `index` over the provided input values, re-entering the active
    /// program interpreter, and returns the region's output values.
    fn interpret_region(&self, context: &C, index: usize, inputs: Vec<C::Value>)
    -> Result<Vec<C::Value>, ProgramError>;

    /// Binds `operation`, together with the provided attached `regions`, through the active interpretation context
    /// and returns its output values. This is the operation-level counterpart of [`Self::interpret_region`]: a
    /// higher-order rule that executes nested work itself instead of replaying one of its regions verbatim (e.g., the
    /// `shard_map` rule, which replays its body once per device and binds each body instruction separately) applies
    /// those instructions through this function. Rules cannot call [`Context::bind`] themselves, because
    /// [`InterpretableOperation`] is bounded by [`Domain`] rather than by [`Context`] (refer to its documentation for
    /// why reaching [`Context`] from a rule would make [`EagerContext`]'s own obligation recursive). The driver is
    /// constructed where the context's [`Context`] implementation is already in scope, so it can bind on the rule's
    /// behalf without that cycle.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation [`Domain`] that the rule received, through which the operation is bound.
    ///   - `operation`: [`Operation`] being applied.
    ///   - `regions`: Application-scoped [`BindingRegionDriver`] providing the complete ordered
    ///     [`Region`](crate::Region)s of this application (e.g., a [`ReplayRegionDriver`] over the attached regions
    ///     of a nested [`Instruction`]).
    ///   - `inputs`: Input values supplied to the operation, in operation-defined order.
    ///
    /// # Errors
    ///
    /// Returns the errors of binding `operation` through the active context, and [`ProgramError::UnsupportedOperation`]
    /// when this driver has no active context to bind through (e.g., [`EmptyRegionDriver`], which is used for direct
    /// rule invocations).
    fn bind<R: BindingRegionDriver<C::Constant, C::Operation>>(
        &self,
        context: &C,
        operation: C::Operation,
        regions: R,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

impl<C: Domain> InterpretationDriver<C> for EmptyRegionDriver {
    #[inline]
    fn interpret_region(
        &self,
        _context: &C,
        _index: usize,
        _inputs: Vec<C::Value>,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot interpret a region".to_string()))
    }

    #[inline]
    fn bind<R: BindingRegionDriver<C::Constant, C::Operation>>(
        &self,
        _context: &C,
        operation: C::Operation,
        _regions: R,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot bind `{}` through an empty region driver, which has no active interpretation context",
                operation.name(),
            ),
        })
    }
}

/// Adapts the [`BindingRegionDriver`] supplied to one eager [`Context::bind`] operation application to the
/// interpretation-specific recursion provided by [`InterpretationDriver`]. Its regions are exactly the nested
/// computations supplied to that application, and [`EagerInterpretationDriver::interpret_region`] replays a
/// selected region through the [`EagerContext`] without taking ownership of it.
pub(crate) struct EagerInterpretationDriver<'r, D> {
    /// Binding [`RegionDriver`] supplied to the active [`Operation`] application.
    driver: &'r D,
}

impl<'r, D> EagerInterpretationDriver<'r, D> {
    /// Creates a new [`EagerInterpretationDriver`].
    #[inline]
    pub(crate) fn new(driver: &'r D) -> Self {
        Self { driver }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for EagerInterpretationDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<
    V: Value,
    O: Operation<Type = V::Type>
        + InterpretableOperation<EagerContext<V, O>>
        + OperationProvider<V::Type, GuardOperation<V::Type>, Operation = O>,
    D: RegionDriver<V, O>,
> InterpretationDriver<EagerContext<V, O>> for EagerInterpretationDriver<'_, D>
{
    #[inline]
    fn interpret_region(
        &self,
        context: &EagerContext<V, O>,
        index: usize,
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError> {
        self.region(index)?.interpret_in_context(context, inputs)
    }

    #[inline]
    fn bind<R: BindingRegionDriver<V, O>>(
        &self,
        context: &EagerContext<V, O>,
        operation: O,
        regions: R,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        context.bind(operation, regions, inputs)
    }
}

// TODO(eaplatanios): Restore the strict `Operation<Type = C::Type>` super-trait bound once the next-generation trait
//  solver stabilizes. The current solver cannot discharge this projection equality at implementation heads whose
//  context type is built from `Self` (E0284), and a per-method `where Self: Operation<Type = C::Type>` clause
//  reproduces the same failure for the composite eager dispatcher for `ArrayIrOperation`.
/// Eager execution rule for an [`Operation`] over a chosen value semantics. The interpretation [`Domain`] `C` is the
/// source of the input, output, and attached-region value families. Implementations add only the capabilities they
/// actually consume, such as zero construction for a nullary rule. The contract requires [`Domain`] rather than
/// [`Context`] because [`EagerContext`]'s [`Context`] implementation itself requires `O: InterpretableOperation<Self>`;
/// reaching [`Context`] through this trait would make that obligation recursive. Rules that need to apply operations
/// themselves bind them through [`InterpretationDriver::bind`] instead.
///
/// # Type Compatibility
///
/// Unlike other operation traits, this contract cannot currently enforce `Self::Type == C::Type`.
/// Its plain [`Operation`] super-trait does not restate the equality, so an implementation for a context whose
/// `C::Type` disagrees with [`Operation::Type`] compiles. The relaxation is forced rather than chosen. The current
/// trait solver cannot discharge that projection equality at implementation heads whose interpretation domain is
/// itself built from `Self`, which is exactly the shape of every eager operation-family dispatcher, and the per-method
/// `where Self: Operation<Type = C::Type>` clause that [`BatchableOperation`](crate::BatchableOperation) and
/// [`DifferentiableOperation`](crate::DifferentiableOperation) use instead reproduces the same failure for the
/// composite eager [`ArrayIrOperation`](crate::ArrayIrOperation) dispatcher.
///
/// Ryft's framework-mediated interpretation paths enforce the missing equality at their surrounding boundaries:
/// [`Program`]s and [`ProgramBuilder`](crate::ProgramBuilder)s require `O: Operation<Type = V::Type>`,
/// the [`Domain::Operation`] item bound pins the domain's own operation family, and the [`InterpretationDriver`]
/// implementations that hand regions to a rule restate `O: Operation<Type = V::Type>` themselves. Families generated
/// by `#[derive(Operation)]` additionally constrain each native payload to the family's primary type and each declared
/// member payload to its member type. Generated member adapters project values into that native type before invoking
/// its eager rule. A mismatched implementation therefore cannot participate in ordinary eager or staged program
/// execution. This does not make such an implementation unreachable. Because this trait and its method are public,
/// downstream code can implement it for a mismatched context and invoke [`interpret`](Self::interpret) directly.
/// Until the strict bound can be restored, direct implementers and callers are responsible for maintaining
/// `Self::Type == C::Type`.
pub trait InterpretableOperation<C: Domain>: Operation {
    /// Interprets this [`Operation`] given the provided input values and returns the resulting output values.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] providing the value-construction capabilities used by this rule.
    ///   - `driver`: [`InterpretationDriver`] providing [`Instruction`]-scoped access to the attached
    ///     application [`Region`](crate::Region)s.
    ///   - `inputs`: Input values to interpret this [`Operation`] on.
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

/// Interpretation rule for a member [`Operation`] whose instruction executes in a different parent value universe.
/// Ordinary [`InterpretableOperation`] implementations describe values of the operation's native type. This trait is
/// reserved for payloads whose parent instruction additionally consumes or produces other member kinds, such as an
/// array constructor whose dynamic extents are first-class dimension values.
pub trait MemberInterpretableOperation<C: Domain>: Operation {
    /// Interprets this payload using values from its enclosing parent [`Domain`].
    ///
    /// # Parameters
    ///
    ///   - `context`: Parent [`Domain`] in which the complete mixed [`Instruction`] executes.
    ///   - `driver`: [`InterpretationDriver`] providing [`Instruction`]-scoped access to the attached
    ///     application [`Region`](crate::Region)s.
    ///   - `inputs`: Parent-universe inputs in [`Instruction`] order.
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError>;
}

impl<
    T: Type,
    V: Value<Type = T>,
    O: Operation<Type = T>,
    Input: Parameterized<V, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V>,
> Program<V, O, Input, Output>
{
    /// Interprets/executes this [`Program`] with the provided input. This is the main replay entry point for staged
    /// [`Program`]s. It checks that the provided input value matches the program's expected input structure and type,
    /// evaluates the [`Instruction`]s in order, and finally builds a structured output value from the computed output
    /// values. This is the eager instantiation of [`Self::interpret_in_context`] using this program's own
    /// [`EagerContext`], whose [`bind`](Context::bind) interprets each operation immediately through its
    /// [`InterpretableOperation`] rule and whose [`lift`](Context::lift) is the identity.
    #[inline]
    pub fn interpret(&self, input: Input) -> Result<Output, ProgramError>
    where
        O: InterpretableOperation<EagerContext<V, O>> + OperationProvider<T, GuardOperation<T>, Operation = O>,
        Input: Parameterized<V, To<V> = Input>,
        Output: Parameterized<V, To<V> = Output>,
    {
        self.interpret_in_context(&EagerContext::<V, O>::new(), input)
    }

    /// Interprets/executes this [`Program`] with the provided input by replaying it through the supplied [`Context`].
    /// Constants are lifted with [`Context::lift`] and each [`Instruction`] is bound with [`Context::bind`]. The
    /// program stays over the context's staged `Constant` representation `V` while values of the context's `Value`
    /// type flow through the replay, so the input and output are the program's `Input` and `Output` reparameterized
    /// at `C::Value`. Because the context owns the semantics of each bind, this gives the eager/staging duality for
    /// free. An eager context (for which `C::Value = V` and this function is [`Self::interpret`]) computes each
    /// operation immediately through its [`InterpretableOperation`] implementation, a staging context records the
    /// replayed operations into the active trace, and a transform context runs its per-operation rules. It checks that
    /// the provided input matches the program's expected input structure and types before replaying.
    ///
    /// This is the plain-program sibling of [`PartialEvaluation::interpret`](crate::PartialEvaluation::interpret),
    /// which additionally wires residual-input feeders, and the transform-aware counterpart of structural relocation
    /// through [`ProgramBuilder::splice_program`](crate::ProgramBuilder::splice_program), which records
    /// [`Instruction`]s directly into a builder without routing through `bind`'s transform interception. Nested program
    /// interpretation (e.g., control flow branches, custom function programs, etc.) routes back through here via the
    /// driver behind the rule's [`InterpretationDriver`].
    pub fn interpret_in_context<C: Context<Type = T, Constant = V, Operation = O>>(
        &self,
        context: &C,
        input: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, ProgramError>
    where
        Input::Family: ParameterizedFamily<C::Value>,
        Output::Family: ParameterizedFamily<C::Value>,
    {
        // Validate that the caller supplied an input with the expected parameter structure.
        let input_structure = input.parameter_structure();
        if input_structure != self.input_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{:?}", self.input_structure),
                right_structure: format!("{input_structure:?}"),
            }
            .into());
        }

        // Program validates the structured parameters while its entry region owns type validation, and replay.
        let outputs = self.entry_region_ref().interpret_in_context(context, input.into_parameters().collect())?;
        Ok(Output::To::<C::Value>::from_parameters(self.output_structure.clone(), outputs)?)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<V, O, Input, Output>
{
    /// Interprets/executes this [`Program`]'s [`Instruction`]s using the caller-supplied value and error semantics.
    /// Transforms and backends specialize this interpretation by choosing a runtime value type `V`, an error type `E`,
    /// a constant-lifting closure `lift_fn`, and an instruction-interpretation closure `interpret_fn`. Inputs and
    /// outputs are flat [`Vec`]s aligned with the program's [`Self::input_ids`] and [`Self::output_ids`]. Structured
    /// input/output handling stays at the call site so that callers can use any parameter family of their choice.
    ///
    /// The `E` type parameter mirrors `V`: a [`Program`] is not tied to a single interpretation, and each
    /// interpretation has its own natural error type. Eager execution interprets instructions into concrete values and
    /// fails with [`ProgramError`], while a backend that lowers each instruction into a compiler IR interprets them
    /// into IR value handles and fails with that backend's own error (e.g., the XLA backend lowers a program into MLIR
    /// values, failing with an MLIR or sharding lowering error). The `E: From<ProgramError>` bound lets one signature
    /// serve both: callers choose the error their closures fail with, and this function's own structural errors (e.g.,
    /// [`ProgramError::UnboundAtomId`] or an input/output count mismatch) fold into that type. A backend error could
    /// instead be boxed into [`ProgramError::Custom`] and recovered by downcasting, but because this function is
    /// already generic over `V`, carrying the matching `E` keeps each interpreter's error statically typed rather than
    /// erasing it to a runtime downcast.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Flat input values aligned with [`Self::input_ids`].
    ///   - `lift_fn`: Closure that lifts an [`Atom::Constant`]'s carried `V` into the runtime leaf type `Value`. This
    ///     closure receives the constant's [`AtomId`] for callers that surface diagnostics or maintain parallel atom
    ///     tables and is invoked at most once per live constant atom, in atom-index order.
    ///   - `interpret_fn`: Closure that interprets one [`Instruction`]'s [`Operation`] to its already-lifted inputs and
    ///     returns the instruction's outputs. The full [`Instruction`] is provided so that the closure can inspect the
    ///     operation's expected output [`Atom`] IDs when needed (e.g., to look up output [`Type`]s).
    #[inline]
    pub fn interpret_with<
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        &self,
        inputs: Vec<RuntimeValue>,
        lift_fn: LiftFn,
        interpret_fn: InterpretFn,
    ) -> Result<Vec<RuntimeValue>, Error> {
        self.entry_region_ref().interpret_with(inputs, lift_fn, interpret_fn)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionRef<'_, V, O> {
    /// Interprets this borrowed [`Region`](crate::Region) through the provided [`Context`] using flat input and output
    /// values. The region and every nested region attached to its [`Instruction`]s are replayed directly from the
    /// source arena. When the provided context stages an unchanged nested region, one replay-scoped mapping preserves
    /// repeated roots and shared descendants in the destination arena.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] that assigns meaning to constant lifting and instruction binding.
    ///   - `inputs`: Flat input values aligned with this region's input atoms.
    #[inline]
    pub fn interpret_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: Vec<C::Value>,
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Eager and symbolic replay share one traversal. Eager values establish their relationships at the boundary,
        // while symbolic replay binds the family's guard to preserve relationships that still need runtime checks.
        let mut interpretation = RegionInterpreter::new(context.clone(), self, inputs)?;
        while !interpretation.is_complete() {
            interpretation = interpretation.step()?;
        }
        interpretation.finish()
    }

    /// Interprets/executes this borrowed [`RegionRef`]'s [`Instruction`]s using the caller-supplied value and error
    /// semantics. This is the borrowed-[`Region`](crate::Region) counterpart of [`Program::interpret_with`]. It replays
    /// the region directly from its source arena without first materializing a standalone [`Program`], while preserving
    /// the same flat input, constant-lifting, instruction-dispatch, and output-gathering behavior.
    #[inline]
    pub fn interpret_with<
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        self,
        inputs: Vec<RuntimeValue>,
        lift_fn: LiftFn,
        mut interpret_fn: InterpretFn,
    ) -> Result<Vec<RuntimeValue>, Error> {
        let mut interpretation = RegionInterpreterState::new(self, inputs, lift_fn)?;
        while !interpretation.is_complete() {
            interpretation = interpretation.step(&mut interpret_fn)?;
        }
        interpretation.finish().map_err(Into::into)
    }
}

/// [`TypeIdentity`](crate::TypeIdentity) bookkeeping shared by context interpretation and partial-evaluation replay.
/// Source input facts validate instruction-local reifications; the shared mapping follows replayed definitions and
/// freshens staged constants. Callers choose staging explicitly, because partial evaluation emits residual values
/// even with an eager known-side parent. Entry validation and guard emission remain the calling interpreter's
/// responsibility.
pub(crate) struct ReplayTypeIdentities<T: Type> {
    /// Original source input facts, used to validate reified boundary dimensions before local substitutions.
    input_refinements: T::Refinements,

    /// Declared identities of the source region whose extents are fixed by its reference inputs: the identities in
    /// the declared types of the reference inputs, plus every declared identity that the entry facts place in the
    /// same class as one of them. For example, a region declaring inputs `ref<f32[n]>` and `dimension<k>` that is
    /// entered with `ref<f32[a]>` and `dimension<a>` records `n` and `k`, since both are tied to the referent's
    /// dimension `a`. A reference's referent type can never be retyped, so an instruction operand whose declared
    /// type names one of these identities is checked against the reference's dimensions, not the other way
    /// around (refer to [`Self::instruction_guard`]).
    reference_declared_identities: Vec<T::Identity>,

    /// Actual identities carried by the supplied reference inputs (i.e., the identities of their referent types; `a`
    /// in the example above). The guards that [`Self::instruction_guard`] builds keep these identities on their
    /// outputs, so a checked ordinary operand adopts the reference's name for a shared dimension instead of a static
    /// extent or another identity. The reference's type is fixed, and later reads of it must still agree with that
    /// operand (refer to [`GuardOperation::retaining`]).
    reference_actual_identities: Vec<T::Identity>,

    /// Source-to-replayed identity substitutions shared by constants, operations, attached regions, and outputs.
    identity_renaming: RefCell<TypeIdentityRenaming<T::Identity>>,
}

impl<T: Type> ReplayTypeIdentities<T> {
    /// Creates the bookkeeping for one replay of `region` whose inputs have already been validated against the declared
    /// input types (`refinements` are the facts that validation established). The bookkeeping starts with an empty
    /// identity renaming when the replay executes eagerly, because eager values carry exact types and the boundary
    /// facts describe them completely. When the replay stages a program instead (i.e., when `staging` is `true`), the
    /// renaming starts with two kinds of substitutions, so that the staged program names the identities of the values
    /// it receives rather than the source region's own:
    ///
    ///   - A dimension input that defines an identity is mapped to the identity of the value supplied for it. For
    ///     example, a pullback declared to return `f32[n, 4]` may receive its dimension residual as `dimension<n2>`,
    ///     and the mapping `n -> n2` makes the staged output refer to that residual. Only inputs with a definition
    ///     position take part; an array such as `f32[n]` cannot establish such a mapping.
    ///   - Every identity that a constant of the region defines is mapped to a fresh identity that no input and no
    ///     other identity of the region uses. Two replays of the same region into one destination (e.g., a primal
    ///     and its pullback) would otherwise both define the source's literal dimension `2`, and each definition
    ///     must be unique in the destination while staying shared by all uses within its own replay.
    ///
    /// # Parameters
    ///
    ///   - `region`: Source region being replayed.
    ///   - `input_types`: Types of the values that enter the region, after any entry guard prepared them.
    ///   - `refinements`: Facts established by validating `input_types` against the region's declared input types,
    ///     kept to validate reified boundary dimensions at individual instructions.
    ///   - `staging`: Whether the replay stages a program rather than executing eagerly. Partial evaluation passes
    ///     `true` even with an eager known-side parent, because it stages residual values.
    pub(crate) fn new<V: Value<Type = T>, O: Operation<Type = T>>(
        region: RegionRef<'_, V, O>,
        input_types: &[T],
        refinements: T::Refinements,
        staging: bool,
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, region.input_ids().len(), ProgramError);
        let identity_renaming = RefCell::new({
            // Eager kernels materialize concrete shapes and keep the original boundary refinement facts.
            // Only staging substitutes validated formal definitions and freshens constant-owned identities.
            let mut renaming = TypeIdentityRenaming::new();
            if staging {
                let signature = region.type_identity_signature();
                let mut input_definitions = Vec::new();
                for input in region.input_ids() {
                    input_definitions.extend(
                        region.atoms()[input.index()]
                            .r#type()
                            .identities()
                            .filter(|(position, _)| *position == TypeIdentityPosition::Definition)
                            .map(|(_, identity)| identity.clone()),
                    );
                }

                if !input_definitions.is_empty() {
                    // Complete boundary refinement has already succeeded. Dimension-valued inputs can carry a more
                    // precise nominal definition, which subsequent staged array payloads must reference consistently
                    // with those inputs. For example, a pullback declared to return `Array[n, 4]` may receive its
                    // dimension residual as `n2`. Mapping the validated dimension input `n -> n2` makes the output
                    // payload refer to that residual. Array shape references alone cannot establish such a mapping;
                    // only definition positions participate.
                    let (declared, actual): (Vec<_>, Vec<_>) = region
                        .input_types()
                        .into_iter()
                        .zip(input_types)
                        .filter(|(declared, _)| {
                            declared.identities().any(|(position, _)| position == TypeIdentityPosition::Definition)
                        })
                        .map(|(declared, actual)| (declared, actual.clone()))
                        .unzip();
                    for (source, target) in T::derive_identity_renaming(&declared, &actual)?.replacements() {
                        if source != target && input_definitions.contains(source) {
                            renaming.insert(source.clone(), target.clone())?;
                        }
                    }
                }

                if !signature.internal_identities().is_empty() {
                    // Two retained programs can lift the same literal into one destination. Its local definition must
                    // be fresh for each replay, while identities belonging to input bindings retain their established
                    // type-identity renaming. For example, replaying a primal and its pullback must not define the same
                    // source dimension constant `2` twice in the destination. Each replay gets its own definition,
                    // shared consistently by all its local uses.
                    let mut unavailable = None;
                    for atom in region.atoms() {
                        if let Some(constant) = atom.as_constant() {
                            for (position, identity) in constant.r#type().identities() {
                                if position == TypeIdentityPosition::Definition
                                    && signature.internal_identities().contains(identity)
                                    && !renaming.replacements().iter().any(|(source, _)| source == identity)
                                {
                                    let unavailable = unavailable.get_or_insert_with(|| {
                                        let mut identities = signature.identities().to_vec();
                                        for input_type in input_types {
                                            identities
                                                .extend(input_type.identities().map(|(_, identity)| identity.clone()));
                                        }
                                        identities
                                    });
                                    let target = renaming.insert_fresh(identity.clone(), unavailable)?;
                                    unavailable.push(target);
                                }
                            }
                        }
                    }
                }
            }
            renaming
        });

        let mut reference_declared_identities = region
            .input_types()
            .iter()
            .filter(|r#type| r#type.is_reference())
            .flat_map(|r#type| r#type.identities().map(|(_, identity)| identity.clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for class in refinements.identity_classes() {
            if class.declared().iter().any(|identity| reference_declared_identities.contains(identity)) {
                reference_declared_identities.extend(class.declared().iter().cloned());
            }
        }

        let reference_actual_identities = input_types
            .iter()
            .filter(|r#type| r#type.is_reference())
            .flat_map(|r#type| r#type.identities().map(|(_, identity)| identity.clone()).collect::<Vec<_>>())
            .collect();

        Ok(Self {
            input_refinements: refinements,
            reference_declared_identities,
            reference_actual_identities,
            identity_renaming,
        })
    }

    /// Returns the guard that `instruction`'s operands must pass through before the instruction is replayed, together
    /// with the positions (in increasing order) of the operands it takes, or [`None`] when no guard is needed. The
    /// guard takes every operand whose declared type shares a dimension identity with a reference input (refer to
    /// [`Self::reference_declared_identities`]) and declares those operands' source types, retaining the identities
    /// of the supplied references (refer to [`GuardOperation::retaining`]). No guard is needed when no operand is
    /// related to a reference input, when the related operands already have exactly their declared types, or when
    /// the guard would leave every non-reference operand's type unchanged. The caller binds the returned guard over
    /// the operands at the returned positions and replaces each non-reference one of them with the corresponding
    /// guard output before binding the instruction.
    ///
    /// The checked values, rather than captured handles, witness the relationship, so a preceding freeze or discharge
    /// cannot invalidate the guard. Ordinary broadcasting outside those relationships remains the consuming operation's
    /// responsibility. Entry exact facts are not applied here: each guard establishes only its own current operand
    /// relationship.
    ///
    /// # Parameters
    ///
    ///   - `region`: Source region whose instruction is being replayed.
    ///   - `instruction`: Source instruction about to be replayed.
    ///   - `input_types`: Types of the replayed values supplied as the instruction's operands, in operand order.
    pub(crate) fn instruction_guard<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        region: RegionRef<'_, V, O>,
        instruction: &Instruction<O>,
        input_types: &[T],
    ) -> Result<Option<(GuardOperation<T>, Vec<usize>)>, TypeError> {
        if self.reference_declared_identities.is_empty() {
            return Ok(None);
        }

        let members = instruction
            .inputs()
            .iter()
            .enumerate()
            .filter_map(|(index, id)| {
                region.atoms()[id.index()]
                    .r#type()
                    .identities()
                    .any(|(_, identity)| self.reference_declared_identities.contains(identity))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if members.is_empty() {
            return Ok(None);
        }

        let declared = members
            .iter()
            .map(|&index| region.atoms()[instruction.inputs()[index].index()].r#type().into_owned())
            .collect::<Vec<_>>();
        let actual = members.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
        if declared == actual {
            return Ok(None);
        }

        let guard = GuardOperation::new(instruction.operation().name(), declared)
            .retaining(&actual, &self.reference_actual_identities)?;
        let outputs = guard.infer_output_types(&actual, &[])?;
        let changed = actual
            .iter()
            .filter(|r#type| !r#type.is_reference())
            .zip(outputs)
            .any(|(actual, output)| *actual != output);
        Ok(changed.then_some((guard, members)))
    }

    /// Returns `constant`, a constant of the source region, with the shared identity renaming applied, so that the
    /// caller lifts a constant whose type names the replay's identities (e.g., the fresh identity of a literal
    /// dimension) rather than the source region's.
    #[inline]
    pub(crate) fn rename_constant<V: Value<Type = T>>(&self, constant: &V) -> Result<V, TypeError> {
        constant.rename_type_identities(&self.identity_renaming.borrow())
    }

    /// Returns the [`TypeIdentityRenaming`] to apply to `instruction`'s operation and attached regions when it is
    /// replayed over operands of types `input_types`: the shared renaming, overridden for this instruction alone
    /// wherever an operand reifies a boundary dimension under a new identity (e.g., `dimension_size` of an input
    /// `f32[q]` replayed at `f32[2]` defines a fresh exact dimension). The override stays local because another
    /// instruction may reify the same dimension under yet another identity; every reification is still validated
    /// against the original boundary facts, so all of them must have the same extent.
    pub(crate) fn instruction_renaming<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        region: RegionRef<'_, V, O>,
        instruction: &Instruction<O>,
        input_types: &[T],
    ) -> Result<TypeIdentityRenaming<T::Identity>, TypeError> {
        let mut renaming = self.identity_renaming.borrow().clone();
        let reified_inputs = instruction
            .inputs()
            .iter()
            .zip(input_types)
            .filter_map(|(id, actual)| {
                let declared = region.atoms()[id.index()].r#type().into_owned();
                let reified = declared.identities().any(|(position, identity)| {
                    position == TypeIdentityPosition::Definition
                        && !region.type_identity_signature().internal_identities().contains(identity)
                });
                reified.then(|| (declared, actual.clone()))
            })
            .collect::<Vec<_>>();

        if !reified_inputs.is_empty() {
            // An input array can establish q without carrying its first-class dimension value. Replaying
            // `size = dimension_size(x); zero[q](size)` at `x: Array[2]` reifies `q` as a fresh exact dimension.
            // Use that explicit input's identity for this constructor and its regions. Keep this local as a second
            // `dimension_size` may reify the same `q` under another fresh identity, and so globally renaming `q` would
            // incorrectly make those two instructions define one identity. Original boundary facts still require every
            // reification of `q` to have the same extent.
            self.input_refinements.validate_refinement(
                reified_inputs.iter().map(|(declared, _)| declared),
                reified_inputs.iter().map(|(_, actual)| actual),
                region.type_identity_signature().identities(),
            )?;
            let local = T::derive_identity_renaming(
                &reified_inputs.iter().map(|(declared, _)| declared.clone()).collect::<Vec<_>>(),
                &reified_inputs.iter().map(|(_, actual)| actual.clone()).collect::<Vec<_>>(),
            )?;
            let mut combined = TypeIdentityRenaming::new();
            for (source, target) in renaming.replacements() {
                if !local.replacements().iter().any(|(local_source, _)| local_source == source) {
                    combined.insert(source.clone(), target.clone())?;
                }
            }
            for (source, target) in local.replacements() {
                combined.insert(source.clone(), target.clone())?;
            }
            renaming = combined;
        }

        Ok(renaming)
    }

    /// Extends the shared identity renaming with the identities that `instruction` defines, mapping each one to the
    /// identity its replayed output of type `output_types` carries, so that later instructions, attached regions, and
    /// the output validation refer to the replayed definitions. Only identities defined inside the source region are
    /// recorded; input identities were mapped at entry and output references introduce no replacements.
    ///
    /// `replayed_input_types` is supplied by partial-evaluation replay only. An output that is one of those inputs
    /// forwarded unchanged (a proven fold) and whose declared type the forwarded type does not refine introduces no
    /// new definition and is skipped: it must keep its own identity rather than be renamed to the source's result
    /// descriptor, which may be narrower (e.g., an arithmetic result capped at the backend width).
    pub(crate) fn record_outputs<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        region: RegionRef<'_, V, O>,
        instruction: &Instruction<O>,
        output_types: &[T],
        replayed_input_types: Option<&[T]>,
    ) -> Result<(), ProgramError> {
        check_count!("output", output_types, instruction.outputs().len(), ProgramError);

        let internal_identities = region.type_identity_signature().internal_identities();
        if internal_identities.is_empty() {
            return Ok(());
        }

        let (declared, actual): (Vec<_>, Vec<_>) = instruction
            .outputs()
            .iter()
            .map(|id| region.atoms()[id.index()].r#type().into_owned())
            .zip(output_types)
            .filter(|(declared, actual)| {
                // A proven fold can forward an unbounded dimension while the source arithmetic result carries a
                // finite backend-width cap. Its input identity already exists, so partial replay must not impose
                // the old result descriptor on that forwarded value or rename metadata to an inadmissible alias.
                !(replayed_input_types.is_some_and(|inputs| inputs.contains(actual))
                    && declared.identities().any(|(position, identity)| {
                        position == TypeIdentityPosition::Definition && internal_identities.contains(identity)
                    })
                    && !declared.is_refined_by(actual))
            })
            .map(|(declared, actual)| (declared, actual.clone()))
            .unzip();

        let definitions = declared
            .iter()
            .flat_map(Type::identities)
            .filter(|(position, identity)| {
                *position == TypeIdentityPosition::Definition && internal_identities.contains(identity)
            })
            .map(|(_, identity)| identity.clone())
            .collect::<Vec<_>>();
        if definitions.is_empty() {
            return Ok(());
        }

        let output_renaming = T::derive_identity_renaming(&declared, &actual)?;
        let mut renaming = self.identity_renaming.borrow_mut();
        for (source, target) in output_renaming.replacements() {
            if source != target && definitions.contains(source) {
                // An unchanged definition adds no type-identity replacement. A later read of that same source
                // dimension may refine its bounds and establish the first nontrivial replacement.
                renaming.insert(source.clone(), target.clone())?;
            }
        }

        Ok(())
    }
}

/// Resumable context-driven interpreter with the same boundary refinements, provenance, identity renaming, and
/// nested-region driver as [`RegionRef::interpret_in_context`]. This wraps [`RegionInterpreterState`] to retain
/// context-specific validation and binding between steps. Each step completes one outer instruction; an attached
/// region executes through the context's ordinary operation rule within that step.
pub(crate) struct RegionInterpreter<'r, C: Context> {
    /// [`Context`] used to lift source constants and bind instructions to runtime values or staged operations. The same
    /// context is retained across every [`Self::step`], so context-owned invocation state and effect handling survive
    /// suspension. Nested regions are interpreted or staged through its ordinary binding rules and the instruction's
    /// [`ReplayRegionDriver`]; [`Self::state`] manages instruction progress independently of those rules.
    context: C,

    /// Borrowed source region together with its resumable instruction position, runtime atom values, and remaining-use
    /// counts. The region's atom types, boundary identifiers, and type-identity signature define the contract checked
    /// by this interpreter. Each [`Self::step`] supplies context-aware dispatch to this state, which gathers inputs
    /// and retains live results. Refinement checks, identity substitutions, and attached-region mappings stay in this
    /// interpreter so ordinary replay and scheduler-selected steps share the same traversal and value-transfer rules.
    state: RegionInterpreterState<'r, C::Constant, C::Operation, C::Value>,

    /// Facts established at entry by validating the supplied input types against the declared input types, expressed
    /// under the entry identity renaming (e.g., an actual static extent refining a declared symbolic dimension).
    /// [`Self::finish`] validates the declared output types, renamed the same way, against the actual outputs with
    /// these facts, so an extent learned from an input also constrains every output that refers to that dimension.
    refinements: <C::Type as Type>::Refinements,

    /// Identity bookkeeping of this [`RegionInterpreter`] which contains the original entry facts that validate reified
    /// boundary dimensions, the identities tied to reference inputs whose operands [`Self::step`] guards, and the
    /// shared renaming from source identities to the identities of replayed values, which constants, operations
    /// attached regions, and the final output validation all apply.
    type_identities: ReplayTypeIdentities<C::Type>,

    /// Replay-scoped import mappings and renamed-source caches shared by the [`ReplayRegionDriver`]s created in
    /// [`Self::step`]. Repeated attached roots and shared descendants retain their identity when imported into the
    /// same destination builder; mappings for different builders remain separate. These [`RegionId`](crate::RegionId)
    /// mappings describe where regions were imported, while [`Self::type_identities`] describes substitutions inside
    /// types and operation payloads. The mappings belong to this source-arena replay and must not be reused for another
    /// arena.
    region_mappings: RegionReplayMappings<C::Constant, C::Operation>,
}

impl<'r, C: Context> RegionInterpreter<'r, C> {
    /// Creates a new [`RegionInterpreter`] by validating input arity and source identifiers, then establishing
    /// refinement facts for the complete input signature. When staging, input and constant-owned type identities are
    /// renamed before live constants are lifted through the context. The resulting [`Self::state`] retains the inputs
    /// and constants needed by later instructions or region outputs; no instruction has been dispatched when this
    /// returns.
    ///
    /// Construction can fail during boundary validation, identity renaming, or constant lifting. An empty instruction
    /// list is already complete, but [`Self::finish`] is still required to gather and validate its outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context retained for constant lifting, instruction binding, and attached-region interpretation.
    ///   - `region`: Borrowed source region and arena used for replay and boundary validation.
    ///   - `inputs`: Runtime values in the order of the region's input atoms. Their types must satisfy the declared
    ///     input signature, including constraints shared across multiple inputs.
    pub(crate) fn new(
        context: C,
        region: RegionRef<'r, C::Constant, C::Operation>,
        mut inputs: Vec<C::Value>,
    ) -> Result<Self, ProgramError> {
        // Both direct region calls and structured Program calls arrive here with flat values.
        // Check their count before validating types or binding instructions.
        check_count!("input", inputs, region.input_ids().len(), ProgramError);

        // Validate boundary identifiers before refinement handling indexes the atom table. Sealed regions normally
        // guarantee these identifiers, but malformed internal state must still produce an error rather than a panic.
        let input_ids = region.input_ids();
        for input_id in input_ids {
            if input_id.index() >= region.atoms().len() {
                return Err(ProgramError::UnboundAtomId { id: *input_id });
            }
        }

        // Equal boundary types carry no additional refinement facts, so avoid constructing a refinement environment
        // in the common staging/replay case. More precise actual types (e.g., a static extent supplied for a dynamic
        // dimension) must be checked as one complete signature so repeated identities agree across inputs.
        let refinements = if input_ids.len() == inputs.len()
            && input_ids
                .iter()
                .zip(&inputs)
                .all(|(id, actual)| region.atoms()[id.index()].r#type().as_ref() == actual.r#type().as_ref())
        {
            <C::Type as Type>::Refinements::default()
        } else {
            <C::Type as Type>::Refinements::new(
                input_ids.iter().map(|id| region.atoms()[id.index()].r#type()),
                inputs.iter().map(Typed::r#type),
            )
            .map_err(|error| Self::contextualize_refinement_error(region, error, input_ids, &inputs, "input"))?
        };

        // Replay through the context's lift/bind protocol. All instructions share one region mapping scope so that
        // a staging destination imports each unchanged source region at most once. Both public entry points use this
        // loop: structured programs and borrowed nested regions must observe identical replay and refinement rules.
        //
        // Dimension definitions also need type-identity renaming between the source and replayed graphs. For example:
        //
        //   - Source: `m1 = n * 2; reshape(x, [2, m1])`
        //   - Replay: `m2 = n * 2; reshape(x, [2, m2])`
        //
        // Replaying multiplication defines a fresh identity `m2`. Record `m1 -> m2` after binding that instruction,
        // then rename subsequent operation payloads and attached region metadata before binding them. Otherwise, the
        // reshape still refers to `m1`, even though both dimensions may print as `n * 2`. The same type-identity
        // renaming applies to declared output types during validation; it does not equate unrelated dimensions
        // with equal names.
        let region_mappings = RegionReplayMappings::new();
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();

        // A symbolic entry must be checked before strict structural renaming, and its prepared values must enter the
        // body. Reference handles retain their types; ordinary operands can adopt the checked representation. Exact
        // eager values carry their concrete geometry, so the refinement check above already established their
        // relationships and no guard is constructed (refer to [`Context::needs_guards`]).
        if context.needs_guards(&input_types)
            && let Some((guard, members)) = GuardOperation::for_entry("region", &region.input_types(), &input_types)?
        {
            let actual = members.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
            let guard = guard.retaining(&actual, &[])?;
            let guarded = members.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
            let outputs = context.bind(
                C::Operation::provide(guard, &members.iter().map(|&index| &input_types[index]).collect::<Vec<_>>())?,
                Vec::new(),
                &guarded,
            )?;
            check_count!(
                "output",
                outputs,
                members.iter().filter(|&&index| !input_types[index].is_reference()).count(),
                ProgramError,
            );
            let mut outputs = outputs.into_iter();
            for index in members {
                if !input_types[index].is_reference() {
                    inputs[index] = outputs.next().unwrap();
                }
            }
        }

        let type_identities = ReplayTypeIdentities::new(
            region,
            &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
            refinements.clone(),
            !context.is_eager(),
        )?;

        let refinements = if type_identities.identity_renaming.borrow().is_identity() {
            refinements
        } else {
            let declared = region
                .input_types()
                .iter()
                .map(|r#type| r#type.rename_identities(&type_identities.identity_renaming.borrow()))
                .collect::<Result<Vec<_>, _>>()?;
            <C::Type as Type>::Refinements::new(declared.iter(), input_types.iter())?
        };

        let state = RegionInterpreterState::new(region, inputs, |_, constant| {
            context.lift(type_identities.rename_constant(constant)?)
        })?;

        Ok(Self { context, state, refinements, type_identities, region_mappings })
    }

    /// Returns the context used by this interpreter's constant lifting and instruction binding. Schedulers can
    /// inspect its invocation state or perform context-specific checks between steps without replacing the context
    /// or advancing [`Self::state`]. This borrows the retained context rather than creating another replay scope.
    pub(crate) fn context(&self) -> &C {
        &self.context
    }

    /// Returns whether every instruction in the outer region has been dispatched successfully by [`Self::state`].
    /// An attached region is handled within its enclosing instruction's binding, so this does not expose a separate
    /// position inside nested regions. A true result means replay is ready for [`Self::finish`]; output collection
    /// and boundary validation have not yet occurred.
    fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Dispatches the next outer instruction and returns the interpreter positioned after it. [`Self::state`]
    /// supplies the inputs; this interpreter validates reified boundary dimensions, applies the appropriate type
    /// identity substitutions, and binds through [`Self::context`] with the source instruction's provenance and a
    /// [`ReplayRegionDriver`]. Fresh region-owned definitions extend [`Self::type_identities`] before later steps
    /// can use them, while [`Self::region_mappings`] preserves attached-region sharing across bindings.
    ///
    /// A step handles one complete outer binding, including any attached-region work performed by its operation
    /// rule. Stepping an already complete interpreter is an error. The function consumes `self`, so a failed binding
    /// or validation cannot resume from inputs already transferred or repeat effects through the same state. This
    /// interpreter does not roll back effects already performed by the context.
    pub(crate) fn step(self) -> Result<Self, ProgramError> {
        let Self { context, state, refinements, type_identities, region_mappings } = self;
        let region = state.region;
        let state = state.step::<ProgramError, _>(|instruction, inputs| {
            // Constant lifting and instruction replay share this type-identity renaming. The operation and driver
            // own their renamed metadata, so release the read borrow before binding and recording fresh output
            // identities. Retaining its guard would make the later mutable borrow fail.
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let prepared = if context.needs_guards(&input_types)
                && let Some((guard, members)) = type_identities.instruction_guard(region, instruction, &input_types)?
            {
                let guarded = members.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
                let outputs = context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    context.bind(
                        C::Operation::provide(
                            guard,
                            &members.iter().map(|&index| &input_types[index]).collect::<Vec<_>>(),
                        )?,
                        Vec::new(),
                        &guarded,
                    )
                })?;
                check_count!(
                    "output",
                    outputs,
                    members.iter().filter(|&&index| !input_types[index].is_reference()).count(),
                    ProgramError,
                );
                let mut prepared = inputs.to_vec();
                let mut outputs = outputs.into_iter();
                for index in members {
                    if !input_types[index].is_reference() {
                        prepared[index] = outputs.next().unwrap();
                    }
                }
                Some(prepared)
            } else {
                None
            };

            let inputs = prepared.as_deref().unwrap_or(inputs);
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let renaming = type_identities.instruction_renaming(region, instruction, &input_types)?;
            let operation = instruction.operation().rename_type_identities(&renaming)?;
            let driver = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?
                .with_type_identity_renaming(&renaming)?;

            // Every replayed instruction binds inside its own recorded origin, so a one-to-one rewrite preserves
            // the source provenance exactly, and a one-to-many rewrite attaches it to every generated instruction.
            // This also holds for unknown source provenance as preserving it exactly means that the replay must not
            // absorb ambient transform scopes.
            let outputs = context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                context.bind(operation, driver, inputs)
            })?;

            let output_types = outputs.iter().map(|output| output.r#type().into_owned()).collect::<Vec<_>>();
            type_identities.record_outputs(region, instruction, &output_types, None)?;
            Ok(outputs)
        })?;

        Ok(Self { context, state, refinements, type_identities, region_mappings })
    }

    /// Consumes a completed interpreter and returns values in the source region's output order. [`Self::state`]
    /// gathers those values using its remaining-use counts; this function then checks the declared output contract
    /// using [`Self::type_identities`]. Type identities allowed by the source signature are renamed consistently,
    /// so output validation accepts replayed definitions without admitting unrelated identities.
    ///
    /// This does not dispatch remaining instructions: an incomplete state is rejected. Output refinement failures
    /// are reported with the same boundary diagnostics as input failures in [`Self::new`]. Both success and failure
    /// consume the interpreter and its replay-scoped mappings.
    pub(crate) fn finish(self) -> Result<Vec<C::Value>, ProgramError> {
        // Replayed values commonly retain the program's exact declared output types. Only a count difference or a more
        // precise output type requires validation. That validation reuses facts established from the inputs and permits
        // a previously unbound output identity only when it belongs to this program's closed identity signature
        // (established by the formal inputs or defined by an instruction inside it).
        let Self { context: _context, state, refinements, type_identities, .. } = self;
        let region = state.region;
        let outputs = state.finish()?;
        let output_ids = region.output_ids();
        if output_ids.len() != outputs.len()
            || output_ids
                .iter()
                .zip(&outputs)
                .any(|(id, actual)| region.atoms()[id.index()].r#type().as_ref() != actual.r#type().as_ref())
        {
            // The declared output types and the closed identity set are renamed the same way as the replayed values,
            // so validation accepts replayed definitions without admitting unrelated identities.
            let renaming = type_identities.identity_renaming.borrow();
            let declared = output_ids
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().rename_identities(&renaming))
                .collect::<Result<Vec<_>, _>>()?;
            let closed_identities = region
                .type_identity_signature()
                .identities()
                .iter()
                .map(|identity| renaming.rename(identity))
                .collect::<Vec<_>>();
            refinements
                .validate_refinement(declared.iter(), outputs.iter().map(Typed::r#type), &closed_identities)
                .map_err(|error| Self::contextualize_refinement_error(region, error, output_ids, &outputs, "output"))?;
        }
        Ok(outputs)
    }

    /// Converts a boundary refinement error to [`ProgramError`], preserving structured errors whenever possible.
    /// For a generic [`TypeError::Invalid`] with matching declared and actual counts, the first pairwise type mismatch
    /// receives an input- or output-specific diagnostic. If no such mismatch exists, the original error is preserved,
    /// including failures caused by relationships across multiple otherwise compatible values. [`Self::new`] and
    /// [`Self::finish`] use this function to keep boundary diagnostics consistent.
    ///
    /// # Parameters
    ///
    ///   - `region`: Source region whose atoms supply the declared boundary types.
    ///   - `error`: Error produced while establishing or validating signature refinements.
    ///   - `ids`: Source input or output atom identifiers, in the same order as `actual`.
    ///   - `actual`: Runtime values checked against those declared types.
    ///   - `position`: Boundary label, either `input` or `output`, included in the mismatch diagnostic.
    fn contextualize_refinement_error(
        region: RegionRef<'r, C::Constant, C::Operation>,
        error: TypeError,
        ids: &[AtomId],
        actual: &[C::Value],
        position: &str,
    ) -> ProgramError {
        if matches!(&error, TypeError::Invalid { .. }) && ids.len() == actual.len() {
            for (id, actual) in ids.iter().zip(actual) {
                let declared = region.atoms()[id.index()].r#type();
                let actual = actual.r#type();
                if !declared.is_refined_by(actual.as_ref()) {
                    return TypeError::invalid(format!(
                        "encountered {position} type {actual} which is incompatible with the region's \
                         declared type {declared}",
                    ))
                    .into();
                }
            }
        }
        error.into()
    }
}

/// Resumable interpretation state for a borrowed [`Region`](crate::Region). This is the same atom environment and
/// last-use transfer used by [`RegionRef::interpret_with`] and context-driven kernel scheduling. Constants are lifted
/// once, and each successful step dispatches exactly one instruction. Consuming steps prevent reuse after a dispatcher
/// failure has consumed inputs or performed an effect.
///
/// [`RegionRef::interpret_with`] supplies constant lifting and instruction dispatch through functions that can use
/// arbitrary runtime values and errors without a [`Context`]. [`RegionInterpreter`] wraps this state to add context
/// binding and validation while sharing the same traversal and liveness rules.
struct RegionInterpreterState<'r, V: Value, O: Operation<Type = V::Type>, RuntimeValue> {
    /// Borrowed canonical source region and its attached region arena.
    region: RegionRef<'r, V, O>,

    /// Runtime atom values retained only while a future instruction or output needs them.
    values: Vec<Option<RuntimeValue>>,

    /// Number of unconsumed instruction-input and region-output occurrences for each atom, indexed like
    /// [`Self::values`]. Each instruction input gathered by [`Self::step`] and each output gathered by [`Self::finish`]
    /// decrements its atom's count. Repeated occurrences count separately. A value is cloned while later consumers
    /// remain and moved out of `values` on its last use; instruction results with no consumers are not stored. Counting
    /// region outputs keeps their values available after [`Self::next_instruction`] reaches the end of the instruction
    /// list.
    remaining_uses: Vec<usize>,

    /// Position of the next instruction in [`Self::region`]'s instruction list. It starts at zero and advances
    /// after [`Self::step`] successfully dispatches one instruction, checks its output count, and stores its live
    /// results in [`Self::values`]. Together with those values and [`Self::remaining_uses`], this position allows
    /// execution to resume without replaying earlier instructions or their effects. Reaching the instruction count
    /// makes [`Self::is_complete`] true; [`Self::finish`] must still gather the region's outputs.
    next_instruction: usize,

    /// Reusable buffer holding inputs in the order required by the instruction selected by [`Self::next_instruction`].
    /// Each [`Self::step`] clears the previous contents, then fills this buffer from [`Self::values`], cloning or
    /// moving each input according to [`Self::remaining_uses`]. The dispatcher borrows the resulting slice; its live
    /// results are stored back in `values`. Capacity is reserved for the largest instruction's input count so
    /// successive steps reuse the allocation. Inputs remain here until the next step clears them or this interpretation
    /// state is finished or dropped.
    instruction_inputs: Vec<RuntimeValue>,
}

impl<'r, V: Value, O: Operation<Type = V::Type>, RuntimeValue: Clone> RegionInterpreterState<'r, V, O, RuntimeValue> {
    /// Creates a new [`RegionInterpreterState`] by checking flat input arity, recording atom uses, and lifting live
    /// constants exactly once. Instruction dispatch has not started when this function returns. The caller supplies the
    /// same value semantics as ordinary flat replay.
    fn new<Error: From<ProgramError>, LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>>(
        region: RegionRef<'r, V, O>,
        inputs: Vec<RuntimeValue>,
        mut lift_fn: LiftFn,
    ) -> Result<Self, Error> {
        let atoms = region.atoms();
        let input_ids = region.input_ids();
        let instructions = region.instructions();
        let output_ids = region.output_ids();
        check_count!("input", inputs, input_ids.len(), ProgramError);

        // Count every future consumer of each atom, including final region outputs. These counts let us move each
        // value out on its last use and clone it only when a later consumer still needs it.
        let mut remaining_uses = vec![0usize; atoms.len()];
        for instruction in instructions {
            for input_id in instruction.inputs().iter().copied() {
                let Some(remaining_uses) = remaining_uses.get_mut(input_id.index()) else {
                    return Err(ProgramError::UnboundAtomId { id: input_id }.into());
                };
                *remaining_uses += 1;
            }
        }
        for output_id in output_ids.iter().copied() {
            let Some(remaining_uses) = remaining_uses.get_mut(output_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: output_id }.into());
            };
            *remaining_uses += 1;
        }

        // Store concrete input values in a sparse value table indexed by `AtomId`.
        let mut values = vec![None; atoms.len()];
        for (input_id, input) in input_ids.iter().copied().zip(inputs) {
            let Some(slot) = values.get_mut(input_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: input_id }.into());
            };
            *slot = Some(input);
        }

        // Materialize literal constants that are live. Dead constants can remain unset because no instruction
        // or region output will read them.
        for (atom_index, atom) in atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(lift_fn(AtomId::new(atom_index), value)?);
            }
        }

        let maximum_input_count = instructions.iter().map(|instruction| instruction.inputs().len()).max().unwrap_or(0);
        Ok(Self {
            region,
            values,
            remaining_uses,
            next_instruction: 0,
            instruction_inputs: Vec::with_capacity(maximum_input_count),
        })
    }

    /// Returns whether every instruction has been dispatched successfully. Outputs still belong to this state until
    /// [`Self::finish`] consumes it, including for a region with no instructions.
    fn is_complete(&self) -> bool {
        self.next_instruction == self.region.instructions().len()
    }

    /// Dispatches exactly one instruction with the supplied semantics and validates its output arity. This consumes
    /// the state so an error cannot resume with inputs already transferred or an effect already executed.
    fn step<
        Error: From<ProgramError>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        mut self,
        mut interpret_fn: InterpretFn,
    ) -> Result<Self, Error> {
        let instruction = self.region.instructions().get(self.next_instruction).ok_or_else(|| {
            ProgramError::MalformedProgram("cannot step a completed region interpretation".to_owned())
        })?;
        self.instruction_inputs.clear();
        for input_id in instruction.inputs().iter().copied() {
            // Consume the appropriate input value for the current instruction. If this is the last consumer,
            // move the value out of the table. Otherwise, clone it so later consumers can still read it.
            let remaining_uses = self.remaining_uses.get_mut(input_id.index()).unwrap();
            debug_assert!(*remaining_uses > 0);
            *remaining_uses -= 1;
            let value = self.values.get_mut(input_id.index()).unwrap();
            let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
            self.instruction_inputs.push(value);
        }

        // Apply the operation using the supplied dispatcher and ensure it produces the expected number of outputs.
        let outputs = interpret_fn(instruction, self.instruction_inputs.as_slice())?;
        check_count!("output", outputs, instruction.outputs().len(), ProgramError);

        for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
            let Some(value) = self.values.get_mut(output_id.index()) else {
                return Err(ProgramError::UnboundAtomId { id: output_id }.into());
            };

            // Keep only outputs with a future consumer. Dead instruction results do not need to occupy the table.
            if self.remaining_uses[output_id.index()] != 0 {
                *value = Some(output);
            }
        }
        self.next_instruction += 1;
        Ok(self)
    }

    /// Gathers flat outputs using the ordinary last-use transfer rules. An incomplete state is rejected instead
    /// of exposing intermediate results as completed program outputs.
    fn finish(mut self) -> Result<Vec<RuntimeValue>, ProgramError> {
        if !self.is_complete() {
            return Err(ProgramError::MalformedProgram(format!(
                "cannot finish region interpretation with {} instructions remaining",
                self.region.instructions().len() - self.next_instruction,
            )));
        }

        // Gather the region outputs using the same last-use transfer logic that we used for instruction inputs.
        let mut outputs = Vec::with_capacity(self.region.output_ids().len());
        for output_id in self.region.output_ids().iter().copied() {
            let remaining_uses = self.remaining_uses.get_mut(output_id.index()).unwrap();
            debug_assert!(*remaining_uses > 0);
            *remaining_uses -= 1;
            let value = self.values.get_mut(output_id.index()).unwrap();
            let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
            outputs.push(value);
        }

        Ok(outputs)
    }
}

/// Interprets a [`Region`](crate::Region)-free member [`Operation`] through a composite [`Domain`]. Inputs are
/// projected to `T`, the payload executes through its native [`EagerContext`], and its outputs are lifted back into the
/// composite value family. This adapter creates neither a second program nor a parallel projected-interpreter protocol.
///
/// This function supports code generated by `#[derive(Operation)]` for `#[ryft(projected(T))]` variants in either
/// transform role. Projected members cannot own composite regions. Mixed and region-carrying operations must keep
/// composite-native interpretation rules.
///
/// # Parameters
///
///   - `context`: Composite domain selecting the input and output value family.
///   - `operation`: Member operation to interpret.
///   - `driver`: Enclosing operation's interpretation driver, used to reject attached composite regions.
///   - `inputs`: Composite input values supplied to the enclosing operation.
pub fn interpret_projected_operation<
    C: Domain<Value: ValueProjection<T, Projected: Value<Type = T>>>,
    T: Type,
    O: Operation<Type = T> + InterpretableOperation<EagerContext<<C::Value as ValueProjection<T>>::Projected, O>>,
    D: InterpretationDriver<C>,
>(
    context: &C,
    operation: &O,
    driver: &D,
    inputs: &[C::Value],
) -> Result<Vec<C::Value>, ProgramError> {
    // `_context` is a zero-cost type witness for the outer domain `C`. It selects the composite value family and
    // interpretation-driver contract, while the projected operation itself must execute through its native eager
    // context below.
    let _ = context;
    if !operation.region_slots().is_empty() || driver.region_count() != 0 {
        return Err(ProgramError::MalformedProgram(format!(
            "projected operation `{}` cannot carry regions",
            operation.name(),
        )));
    }
    let context = EagerContext::<<C::Value as ValueProjection<T>>::Projected, O>::new();
    let interpret = |inputs: &[<C::Value as ValueProjection<T>>::Projected]| {
        operation.interpret(&context, &EmptyRegionDriver, inputs)
    };
    let outputs = match inputs {
        [] => interpret(&[]),
        [input] => {
            let inputs = [C::Value::into_projected(input.clone())?];
            interpret(&inputs)
        }
        [left, right] => {
            let inputs = [C::Value::into_projected(left.clone())?, C::Value::into_projected(right.clone())?];
            interpret(&inputs)
        }
        inputs => {
            let inputs = inputs.iter().cloned().map(C::Value::into_projected).collect::<Result<Vec<_>, _>>()?;
            interpret(&inputs)
        }
    }?;
    Ok(outputs.into_iter().map(C::Value::from_projected).collect())
}

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrTypeRefinements, ArrayIrValue, ArrayOperation, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionError, DimensionOperation, DimensionType, DimensionValue,
        DimensionVariable, Shape,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{
        AddOperation, BroadcastOperation, ConditionOperation, DimensionAddOperation, DimensionSizeOperation,
        NegOperation, ZeroOperation,
    };
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::programs::regions::tests::{StructuralIdentity, StructuralOperation, StructuralType};
    use crate::programs::{
        AtomId, ProgramBuilder, ProgramError, Provenance, ProvenanceScope, ReferenceType, RegionId, RegionInterface,
        RegionSlot, TypeError,
    };
    use crate::tests::{TestArrayContext, TestArrayOperation, TestRegionOperation};
    use crate::tracing::TracingContext;

    use super::*;

    /// Two dependent operations with a shared final result for resumable last-use tests.
    fn stepping_program() -> Program<Array, TestArrayOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let constant = builder.add_constant(Array::scalar(3f32).unwrap());
        builder.add_constant(Array::scalar(99f32).unwrap());
        let sum = builder.add_instruction(AddOperation::new(), vec![], vec![input, constant], None).unwrap()[0];
        let output = builder.add_instruction(NegOperation::new(), vec![], vec![sum], None).unwrap()[0];
        builder.build(vec![output, output], vec![Placeholder], vec![Placeholder; 2]).unwrap()
    }

    #[test]
    fn test_empty_region_driver_interpret_region() {
        let context = TestArrayContext::new();
        let expected = ProgramError::MalformedProgram("empty region driver cannot interpret a region".to_string());
        let expected_region = ProgramError::MalformedProgram("region index 0 is out of range".to_string());
        assert_eq!(RegionDriver::<Array, TestArrayOperation>::regions(&EmptyRegionDriver).count(), 0);
        assert_eq!(RegionDriver::<Array, TestArrayOperation>::region_count(&EmptyRegionDriver), 0);
        assert!(matches!(
            RegionDriver::<Array, TestArrayOperation>::region(&EmptyRegionDriver, 0),
            Err(error) if error == expected_region,
        ));
        assert_eq!(EmptyRegionDriver.interpret_region(&context, 0, Vec::<Array>::new()), Err(expected));
    }

    #[test]
    fn test_empty_region_driver_bind() {
        // An empty region driver serves direct rule invocations, which have no active interpretation context to bind
        // through.
        let context = TestArrayContext::new();
        let input = Array::scalar(2.0f32).unwrap();
        assert_eq!(
            EmptyRegionDriver.bind(&context, AddOperation::new().into(), Vec::new(), &[input.clone(), input]),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot bind `add` through an empty region driver, which has no active interpretation \
                          context"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_eager_interpretation_driver_bind() {
        // The eager driver binds through the eager context that it serves, which interprets the operation.
        let context = TestArrayContext::new();
        let regions = Vec::<Program<Array, TestArrayOperation, Vec<Array>, Vec<Array>>>::new();
        let driver = EagerInterpretationDriver::new(&regions);
        let input = Array::scalar(2.0f32).unwrap();
        assert_eq!(
            driver.bind(&context, AddOperation::new().into(), Vec::new(), &[input.clone(), input]),
            Ok(vec![Array::scalar(4.0f32).unwrap()]),
        );
        assert_eq!(
            driver.bind(&context, AddOperation::new().into(), Vec::new(), &[Array::scalar(2.0f32).unwrap()]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
    }

    #[test]
    fn test_program_interpret_materializes_duplicate_outputs() {
        // A program whose two outputs are the same atom materializes that value into both output positions.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F32));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder
            .build::<Array, (Array, Array)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        assert_eq!(
            program.interpret(Array::scalar(2.0f32).unwrap()),
            Ok((Array::scalar(4.0f32).unwrap(), Array::scalar(4.0f32).unwrap()))
        );
    }

    #[test]
    fn test_program_interpret_in_context_preserves_replayed_region_sharing() {
        let mut nested_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested_input = nested_builder.add_input(ArrayType::scalar(DataType::F64));
        let nested = nested_builder
            .build::<Vec<Array>, Vec<Array>>(vec![nested_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut source_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let shared = source_builder.import_program(nested);
        let source_input = source_builder.add_input(ArrayType::scalar(DataType::F64));
        let first = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![shared],
                vec![source_input],
                None,
            )
            .unwrap()[0];
        let second = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] }),
                vec![shared],
                vec![first],
                None,
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Array>, Vec<Array>>(vec![second], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let context = TracingContext::<Array, TestRegionOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        let outputs = source.interpret_in_context(&context, vec![input]).unwrap();
        let destination = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0].atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(destination.regions().len(), 2);
        assert_eq!(destination.instructions().len(), 2);
        assert_eq!(destination.instructions()[0].regions(), destination.instructions()[1].regions());
        assert_eq!(destination.instructions()[0].regions(), &[RegionId::new(0)]);
    }

    #[test]
    fn test_program_interpret_in_context_preserves_instruction_provenance() {
        // A replay binds each instruction inside its own recorded origin, so a one-to-one rewrite must preserve the
        // source provenance exactly, for nested scope, fused, and unknown source provenance alike.
        let nested = Provenance::scope(
            ProvenanceScope::new("outer"),
            Provenance::scope(ProvenanceScope::new("inner"), Provenance::unknown()),
        );
        let fused = Provenance::fused([
            Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
            Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
        ]);
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let source_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder
            .add_instruction(NegOperation::new(), Vec::new(), vec![source_input], Some(nested.clone()))
            .unwrap()[0];
        let summed = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![negated, source_input], Some(fused.clone()))
            .unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![summed, summed], None).unwrap()[0];
        let source =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // Replays the source program into a fresh trace, either directly or with `outer` entered as an ambient scope,
        // and additionally stages one instruction that is not a replay of any source instruction.
        let replay = |ambient_scope: bool| {
            let context = TracingContext::<Array, TestArrayOperation>::new();
            let input = context.input(ArrayType::scalar(DataType::F64));
            let stage = || {
                let outputs = source.interpret_in_context(&context, vec![input]).unwrap();
                context.bind(NegOperation::new(), Vec::new(), &[outputs[0].clone()]).unwrap();
            };
            match ambient_scope {
                true => context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), stage),
                false => stage(),
            }
            context
                .builder()
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            replay(false),
            vec![
                ("neg", nested.clone()),
                ("add", fused.clone()),
                ("add", Provenance::unknown()),
                ("neg", Provenance::unknown()),
            ],
        );

        // Entering the same scope as the first source instruction records around the replay changes nothing about the
        // replayed provenance, because scopes entered before an origin are the enclosing transform's ambient context
        // and never fold over it. Only the directly staged instruction, which replays nothing, receives that scope.
        assert_eq!(
            replay(true),
            vec![
                ("neg", nested),
                ("add", fused),
                ("add", Provenance::unknown()),
                ("neg", Provenance::scope(ProvenanceScope::new("outer"), Provenance::unknown())),
            ],
        );
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(Array::scalar(7.0f64).unwrap());
        let c1 = builder.add_constant(Array::scalar(3.0f64).unwrap());
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c1], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![Array::scalar(2.0f64).unwrap()],
                |atom_id, value| {
                    lifted_constants.push((atom_id, value.clone()));
                    Ok(value.clone())
                },
                |instruction, inputs| instruction.operation().interpret(
                    &TestArrayContext::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Ok(vec![Array::scalar(5.0f64).unwrap()]),
        );
        assert_eq!(lifted_constants, vec![(c1, Array::scalar(3.0f64).unwrap())]);
        assert_eq!(c0, AtomId::new(1));
    }

    #[test]
    fn test_program_interpret_with_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Vec<Array>, Array>(vec![i0], vec![Placeholder], Placeholder).unwrap();
        assert!(matches!(
            program.interpret(vec![Array::scalar(1.0f64).unwrap(), Array::scalar(2.0f64).unwrap()]),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Placeholder])
                && right_structure
                    == format!("{:?}", vec![Array::scalar(1.0f64).unwrap(), Array::scalar(2.0f64).unwrap()].parameter_structure())
        ));
    }

    #[test]
    fn test_program_interpret_input_type_checking() {
        // A statically typed program input rejects values whose concrete types do not match it exactly.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0]).unwrap()).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message
                == "encountered input type f64[3] which is incompatible with the region's declared type f64[2]",
        ));

        // An unbounded dynamically sized program input accepts concrete values of any size, so one staged program
        // replays at several concrete sizes. Rank mismatches are still rejected.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        ));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0]).unwrap()).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert_eq!(
            program.interpret(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()).unwrap().to_f64s(),
            vec![2.0, 4.0, 6.0]
        );
        assert!(matches!(
            program.interpret(Array::scalar(1.0).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message
                == "encountered input type f64[] which is incompatible with the region's declared type f64[dynamic]",
        ));

        // A bounded dynamically sized program input enforces its exclusive upper bound on concrete sizes.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "dynamic",
                DimensionBounds::non_negative(Some(3)).unwrap(),
            ))]),
        ));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0]).unwrap()).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>()
                    == Some(&DimensionError::BindingOutOfBounds {
                        variable: "dynamic".to_string(),
                        value: 3,
                        bounds: DimensionBounds::non_negative(Some(3)).unwrap(),
                    }),
        ));
    }

    #[test]
    fn test_program_interpret_with_rejecting_guard_provider() {
        // Eager replay never requests guards, so a family whose provider rejects every guard request still replays a
        // region-carrying operation over a symbolic extent, and the branch entries still check the runtime extents.
        let extent = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.simplified_dimension()]));
        let branches = [0, 1].map(|output_index| {
            let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
            let inputs = [builder.add_input(vector_type.clone()), builder.add_input(vector_type.clone())];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![inputs[output_index]], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        });
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let first = builder.add_input(vector_type.clone());
        let second = builder.add_input(vector_type);
        let branches = branches.into_iter().map(|branch| builder.import_program(branch)).collect();
        let outputs = builder
            .add_instruction(ConditionOperation::new(), branches, vec![predicate, first, second], None)
            .unwrap()
            .to_vec();
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let first = Array::vector(vec![1f32, 2.0]).unwrap();
        let second = Array::vector(vec![3f32, 4.0]).unwrap();
        assert_eq!(
            program.interpret(vec![Array::scalar(true).unwrap(), first.clone(), second.clone()]),
            Ok(vec![first.clone()])
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(false).unwrap(), first.clone(), second]),
            Ok(vec![Array::vector(vec![3f32, 4.0]).unwrap()])
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(true).unwrap(), first, Array::vector(vec![1f32, 2.0, 3.0]).unwrap()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_owned(),
                expected: 2,
                actual: 3,
            }))),
        );
    }

    #[test]
    fn test_program_interpret_in_context_guards_unproven_entry_relationships() {
        // Replaying `[f32[n], f32[n]]` at a lifted `f32[4]` and a symbolic `f32[n]` checks only types at trace time,
        // which admit the mixed refinement. The replay stages an entry guard ahead of the body, so the traced program
        // rejects a mismatching runtime extent exactly like direct interpretation.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = builder.add_input(vector_type.clone());
        builder.add_input(vector_type.clone());
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![first],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let four = ArrayIrValue::Array(Array::vector(vec![2f32; 4]).unwrap());
        let three = ArrayIrValue::Array(Array::vector(vec![1f32; 3]).unwrap());
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 4,
            actual: 3,
        })));
        assert_eq!(program.interpret(vec![four.clone(), three.clone()]), mismatch);

        let outer = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = vec![outer.lift(four.clone()).unwrap(), outer.input(vector_type)];
        let outputs = program.interpret_in_context(&outer, inputs).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let traced = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                output_atoms,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {r#"
                lambda %1:f32[n] .
                let %0:f32[4] = const [2.0, 2.0, 2.0, 2.0]
                    %2:f32[4], %3:f32[4] = guard [boundary="region", types=[f32[n], f32[n]]] %0 %1
                in (%2)"#},
        );
        assert_eq!(traced.interpret(vec![four.clone()]), Ok(vec![four]));
        assert_eq!(traced.interpret(vec![three]), mismatch);
    }

    #[test]
    fn test_program_interpret_boundary_refinements() {
        // This sentinel deliberately violates its inferred output shape to exercise replay validation.
        // Valid production operations in `TestArrayOperation` must not implement that behavior.
        #[derive(Clone)]
        struct WrongShapeOperation;

        impl Operation for WrongShapeOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "wrong_shape"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let [input_type] = input_types else {
                    return Err(TypeError::invalid(format!(
                        "wrong shape expects 1 input but got {}",
                        input_types.len(),
                    )));
                };
                Ok(vec![input_type.clone()])
            }
        }
        impl OperationProvider<ArrayType, GuardOperation<ArrayType>> for WrongShapeOperation {
            type Operation = Self;

            fn provide(request: GuardOperation<ArrayType>, _input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
                Err(request.unsupported_family_error())
            }
        }

        impl InterpretableOperation<EagerContext<Array, WrongShapeOperation>> for WrongShapeOperation {
            fn interpret<D: InterpretationDriver<EagerContext<Array, WrongShapeOperation>>>(
                &self,
                _context: &EagerContext<Array, WrongShapeOperation>,
                _driver: &D,
                _inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                Ok(vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap()])
            }
        }

        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let batch = DimensionVariable::new("batch", bounds);
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone())]));

        // Repeated occurrences of one input identity must refine to the same concrete extent across the complete
        // boundary, rather than being checked independently.
        let mut builder = ProgramBuilder::<Array, WrongShapeOperation>::new();
        let first = builder.add_input(dynamic_type.clone());
        builder.add_input(dynamic_type.clone());
        let program = builder
            .build::<(Array, Array), Array>(vec![first], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert!(matches!(
            program.interpret((Array::vector(vec![1.0, 2.0]).unwrap(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap())),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>()
                    == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 2,
                        actual: 3,
                    }),
        ));

        // Output validation reuses the input refinement environment, so an operation that violates its declared
        // identity relationship is rejected even though its output independently refines the dynamic array type.
        let mut builder = ProgramBuilder::<Array, WrongShapeOperation>::new();
        let input = builder.add_input(dynamic_type);
        let output = builder.add_instruction(WrongShapeOperation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret(Array::vector(vec![1.0, 2.0]).unwrap()),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>()
                    == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 2,
                        actual: 3,
                    }),
        ));
        assert!(matches!(
            program
                .entry_region_ref()
                .interpret_in_context(
                    &EagerContext::<Array, WrongShapeOperation>::new(),
                    vec![Array::vector(vec![1.0, 2.0]).unwrap()],
                ),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>()
                    == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 2,
                        actual: 3,
                    }),
        ));

        // Both public entry points use the same output validation and region diagnostics. Use a
        // static declaration here so this is the ordinary type-mismatch diagnostic rather than a refinement conflict.
        let mut builder = ProgramBuilder::<Array, WrongShapeOperation>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let output = builder.add_instruction(WrongShapeOperation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1_f64, 2.]).unwrap();
        assert_eq!(
            program.interpret(input.clone()),
            Err(TypeError::invalid(
                "encountered output type f64[3] which is incompatible with the region's declared type f64[2]",
            )
            .into()),
        );
        assert_eq!(
            program
                .entry_region_ref()
                .interpret_in_context(&EagerContext::<Array, WrongShapeOperation>::new(), vec![input]),
            Err(TypeError::invalid(
                "encountered output type f64[3] which is incompatible with the region's declared type f64[2]",
            )
            .into()),
        );
    }

    #[test]
    fn test_program_interpret_replayed_internal_identities() {
        // The two internal dimensions have equal printed names and bounds, but different definition sites. Replay
        // must retain that distinction while consistently renaming each subsequent constructor's shape payload.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let one = DimensionValue::constant(1).unwrap();
        let one_type = one.r#type().into_owned();
        let one = builder.add_constant(ArrayIrValue::Dimension(one));
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(8)).unwrap());
            let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
            let size = DimensionSizeOperation::new(&input_type, 0).unwrap();
            let addition = DimensionAddOperation::new(size.output_type(), &one_type).unwrap();
            let input = builder.add_input(input_type.into());
            let size = builder.add_instruction(size, Vec::new(), vec![input], None).unwrap()[0];
            let result = builder
                .add_instruction(DimensionOperation::Add(addition), Vec::new(), vec![size, one], None)
                .unwrap()[0];
            let ArrayIrType::Dimension(result_type) = builder.atoms()[result.index()].r#type().into_owned() else {
                panic!("expected a dimension result");
            };
            let output_type = ArrayType::new(DataType::F64, Shape::new(vec![result_type.dimension().clone()]));
            outputs.push(
                builder
                    .add_instruction(
                        ArrayIrOperation::Zero(ZeroOperation::new(output_type.into())),
                        Vec::new(),
                        vec![result],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        outputs.push(outputs[0]);
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let original = program.to_string();
        let specialized = program
            .clone()
            .specialize(&[
                ArrayType::new_static(DataType::F64, [2]).into(),
                ArrayType::new_static(DataType::F64, [3]).into(),
            ])
            .unwrap();
        // Static inputs make every replayed dimension static, so each output takes the extent of its own definition.
        assert_eq!(
            specialized.output_types(),
            vec![
                ArrayType::new_static(DataType::F64, [3]).into(),
                ArrayType::new_static(DataType::F64, [4]).into(),
                ArrayType::new_static(DataType::F64, [3]).into(),
            ],
        );
        assert_eq!(
            specialized.interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[1_f64, 2.]).unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [3]), &[3_f64, 4., 5.]).unwrap()
                ),
            ]),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [3]), &[0_f64; 3]).unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [4]), &[0_f64; 4]).unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F64, [3]), &[0_f64; 3]).unwrap()
                ),
            ])
        );
        assert_eq!(program.to_string(), original);

        // Replaying two retained graphs into the same destination copies each literal dimension once per replay. The
        // literals are static, so they define no identities that the two replays would need to keep apart.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let inputs = vec![
            context.input(ArrayType::new_static(DataType::F64, [2]).into()),
            context.input(ArrayType::new_static(DataType::F64, [3]).into()),
        ];
        let first = program.interpret_in_context(&context, inputs.clone()).unwrap();
        let second = program.interpret_in_context(&context, inputs).unwrap();
        let destination = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                first.iter().chain(&second).map(|value| value.atom_id().unwrap()).collect(),
                vec![Placeholder; 2],
                vec![Placeholder; 6],
            )
            .unwrap();
        let constants = destination.atoms().iter().filter_map(Atom::as_constant).collect::<Vec<_>>();
        assert_eq!(constants.len(), 2);
        assert!(
            constants.iter().all(|constant| constant.r#type().as_ref()
                == &DimensionValue::constant(1).unwrap().r#type().into_owned().into())
        );

        // A more precise dimension input carries its own nominal definition. Array outputs referring to that
        // dimension must respect its validated refinement, including during eager replay.
        let variable = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(DimensionType::from(variable.clone()).into());
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(variable)]));
        let output = builder
            .add_instruction(ArrayIrOperation::Zero(ZeroOperation::new(output_type)), vec![], vec![input], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            source.interpret(vec![ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F64, [3]), &[0_f64; 3]).unwrap()
            ),])
        );

        // Reifying an input-owned identity also determines its type-identity renaming. The input reference
        // establishes `q`, dimension_size defines its first-class value, and zero must use the replayed definition.
        let variable = DimensionVariable::new("queries", DimensionBounds::new(2, Some(4)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![variable.clone().into(), variable.into()]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let sizes = (0..2)
                .map(|axis| {
                    builder
                        .add_instruction(
                            DimensionSizeOperation::new(&input_type, axis).unwrap(),
                            vec![],
                            vec![input],
                            None,
                        )
                        .unwrap()[0]
                })
                .collect();
            outputs.push(
                builder
                    .add_instruction(
                        ArrayIrOperation::Zero(ZeroOperation::new(input_type.clone())),
                        vec![],
                        sizes,
                        None,
                    )
                    .unwrap()[0],
            );
        }
        let source = builder
            .build::<ArrayIrValue<Array>, Vec<ArrayIrValue<Array>>>(outputs, Placeholder, vec![Placeholder; 2])
            .unwrap();
        for extent in [2, 3] {
            let input_type = ArrayType::new_static(DataType::F64, [extent, extent]);
            let specialized = source.clone().specialize(&[input_type.clone().into()]).unwrap();
            let input =
                ArrayIrValue::Array(Array::from_elements(input_type.clone(), &vec![1_f64; extent * extent]).unwrap());
            let expected =
                ArrayIrValue::Array(Array::from_elements(input_type, &vec![0_f64; extent * extent]).unwrap());
            assert_eq!(specialized.interpret(input), Ok(vec![expected.clone(), expected]));
        }
    }

    #[test]
    fn test_program_interpret_does_not_refine_operation_payloads() {
        // Batching a program with a dynamic per-item dimension stages a homogeneous `broadcast` whose stored output
        // type retains that dynamic extent (refer to `test_program_batched_carries_dynamic_per_item_dimensions`).
        // The same payload is built directly here to verify replay independently of batching.
        let dynamic = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()));
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic.clone()]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), dynamic]));
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(input_type);
        let output = builder
            .add_instruction(BroadcastOperation::new(output_type, vec![1]), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[n] .
                let %1:f64[2, n] = broadcast [output_type=f64[2, n], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Broadcasting resolves inherited output dimensions from the concrete input. Interpretation does not
        // mutate the stored operation payload while computing that concrete result.
        let original_program = program.to_string();
        assert_eq!(
            program.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()]),
        );
        assert_eq!(program.to_string(), original_program);

        // A statically typed payload establishes no boundary facts, so ordinary replay is untouched.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let output = builder
            .add_instruction(
                BroadcastOperation::new(
                    ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
                    vec![1],
                ),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap()]),
        );
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_inputs() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Array, Array>(vec![i0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                Vec::<Array>::new(),
                |_, value| Ok(value.clone()),
                |instruction, inputs| instruction.operation().interpret(
                    &TestArrayContext::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let o0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                vec![Array::scalar(2.0f64).unwrap()],
                |_, value| Ok(value.clone()),
                |_, _| Ok::<Vec<Array>, ProgramError>(Vec::new()),
            ),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_region_ref_interpret_in_context() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let r#type = ArrayType::new_static(DataType::F64, [2]);
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let region = program.entry_region_ref();
        let context = TestArrayContext::new();

        // Borrowed regions use the same execution path as structured programs, with a flat input/output boundary.
        assert_eq!(
            region.interpret_in_context(&context, vec![Array::from_elements(r#type.clone(), &[1_f64, 2.]).unwrap()]),
            Ok(vec![Array::from_elements(r#type, &[-1_f64, -2.]).unwrap()]),
        );

        // Flat arity is checked before replay, and pairwise type errors retain the region diagnostic wording.
        assert_eq!(
            region.interpret_in_context(&context, vec![]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            region.interpret_in_context(&context, vec![Array::scalar(1_f64).unwrap()]),
            Err(TypeError::invalid(
                "encountered input type f64[] which is incompatible with the region's declared type f64[2]",
            )
            .into()),
        );
    }

    #[test]
    fn test_region_ref_interpret_in_context_guards_mixed_dimension_array_entries() {
        // A staged replay whose inputs leave a declared relationship unproven (a symbolic dimension next to a static
        // array that both declare `n`) stages an entry guard over those inputs, so the replayed program checks the
        // runtime extents exactly like direct interpretation and the body consumes the guard's outputs.
        type TestValue = ArrayIrValue<Array>;
        type TestOperation = ArrayIrOperation<Array>;

        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let n = DimensionType::new("n", bounds);
        let m = DimensionType::new("m", bounds);
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(n.clone().into());
        builder.add_input(vector.clone().into());
        let output = builder.add_instruction(ZeroOperation::new(vector), vec![], vec![dimension], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        for supplied in [n, m] {
            let name = supplied.variable().unwrap().name().to_owned();
            let context = TracingContext::<TestValue, TestOperation>::new();
            let inputs =
                vec![context.input(supplied.into()), context.input(ArrayType::new_static(DataType::F32, [2]).into())];
            let outputs = source.entry_region_ref().interpret_in_context(&context, inputs).unwrap();
            let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
            let replayed = context
                .builder()
                .borrow()
                .clone()
                .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            assert_eq!(
                replayed.to_string(),
                formatdoc! {
                    r#"
                        lambda %0:dimension<{name} ∈ [1, 8)>, %1:f32[2] .
                        let %2:dimension<2>, %3:f32[2] = guard [boundary="region", types=[dimension<n ∈ [1, 8)>, f32[n]]] %0 %1
                            %4:f32[2] = zero [type=f32[n]] %2
                        in (%4)"#,
                    name = name,
                },
            );
            let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
            let vector = TestValue::Array(Array::vector(vec![1f32; 2]).unwrap());
            let valid = source.interpret(vec![dimension(2), vector.clone()]);
            assert_eq!(valid, Ok(vec![TestValue::Array(Array::vector(vec![0f32; 2]).unwrap())]));
            assert_eq!(replayed.interpret(vec![dimension(2), vector.clone()]), valid);
            let invalid = source.interpret(vec![dimension(3), vector.clone()]);
            assert_eq!(
                invalid,
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_owned(),
                    expected: 3,
                    actual: 2,
                }))),
            );
            assert_eq!(replayed.interpret(vec![dimension(3), vector]), invalid);
        }
    }
    #[test]
    fn test_replay_type_identities_new() {
        // Eager replay keeps the source identities: exact values carry their geometry in their types. Staged replay
        // maps a dimension input's definition to the identity of the supplied value, so staged payloads refer to the
        // value they received, and leaves identities the inputs keep unchanged.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let n2 = DimensionType::new("n2", n.bounds());
        let vector = |dimension: &DimensionType| -> ArrayIrType {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()])).into()
        };
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dimension = builder.add_input(n.clone().into());
        builder.add_input(vector(&n));
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]))),
                vec![],
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let supplied = vec![n2.clone().into(), vector(&n)];
        let refinements = ArrayIrTypeRefinements::new(region.input_types(), &supplied).unwrap();
        let eager = ReplayTypeIdentities::new(region, &supplied, refinements.clone(), false).unwrap();
        assert!(eager.identity_renaming.borrow().is_identity());
        let staged = ReplayTypeIdentities::new(region, &supplied, refinements, true).unwrap();
        assert_eq!(staged.identity_renaming.borrow().rename(n.variable().unwrap()), n2.variable().unwrap().clone());
        let unchanged = region.input_types();
        let refinements = ArrayIrTypeRefinements::new(&unchanged, &unchanged).unwrap();
        let staged = ReplayTypeIdentities::new(region, &unchanged, refinements, true).unwrap();
        assert!(staged.identity_renaming.borrow().is_identity());
        assert_eq!(
            ReplayTypeIdentities::new(region, &supplied[..1], ArrayIrTypeRefinements::default(), true).map(|_| ()),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
    }

    #[test]
    fn test_replay_type_identities_instruction_guard() {
        // Operands whose declared types share a dimension identity with a reference input are guarded: the guard
        // declares their source types and retains the reference's identity, so a related operand that arrives with a
        // static extent adopts the reference's name instead of the reference adopting the extent. Operands unrelated
        // to any reference, or arriving at exactly their declared types, need no guard.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, a] = ["n", "a"].map(|name| DimensionType::new(name, bounds));
        let vector_type =
            |dimension: &DimensionType| ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let vector = |dimension: &DimensionType| -> ArrayIrType { vector_type(dimension).into() };
        let reference =
            |dimension: &DimensionType| -> ArrayIrType { ReferenceType::new(vector_type(dimension)).into() };
        let program = |with_reference: bool| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            if with_reference {
                builder.add_input(reference(&n));
            }
            let first = builder.add_input(vector(&n));
            let second = builder.add_input(vector(&n));
            let output = builder.add_instruction(AddOperation::new(), vec![], vec![first, second], None).unwrap()[0];
            let input_count = if with_reference { 3 } else { 2 };
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder; input_count],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let static_vector: ArrayIrType = ArrayType::new_static(DataType::F32, [4]).into();

        // A related operand with a static extent is guarded together with the symbolic one, and both outputs carry
        // the reference's identity.
        let program_with_reference = program(true);
        let region = program_with_reference.entry_region_ref();
        let supplied = vec![reference(&a), vector(&a), static_vector.clone()];
        let refinements =
            ArrayIrTypeRefinements::from_local_declarations(region.input_types().iter(), supplied.iter()).unwrap();
        let replay = ReplayTypeIdentities::new(region, &supplied, refinements, true).unwrap();
        let (guard, members) = replay
            .instruction_guard(region, &region.instructions()[0], &supplied[1..])
            .unwrap()
            .expect("operands related to a reference input are guarded");
        assert_eq!(members, vec![0, 1]);
        assert_eq!(guard.boundary(), "add");
        assert_eq!(guard.declared_types(), &[vector(&n), vector(&n)]);
        assert_eq!(guard.retained(), &[a.to_variable()]);
        assert_eq!(guard.infer_output_types(&supplied[1..], &[]), Ok(vec![vector(&a), vector(&a)]));

        // Operands arriving at exactly their declared types need no guard.
        let declared = region.input_types();
        let refinements = ArrayIrTypeRefinements::new(&declared, &declared).unwrap();
        let replay = ReplayTypeIdentities::new(region, &declared, refinements, true).unwrap();
        assert_eq!(replay.instruction_guard(region, &region.instructions()[0], &declared[1..]), Ok(None));

        // Without a reference input, no operand is related to one.
        let program_without_reference = program(false);
        let region = program_without_reference.entry_region_ref();
        let supplied = vec![vector(&a), static_vector];
        let refinements =
            ArrayIrTypeRefinements::from_local_declarations(region.input_types().iter(), supplied.iter()).unwrap();
        let replay = ReplayTypeIdentities::new(region, &supplied, refinements, true).unwrap();
        assert_eq!(replay.instruction_guard(region, &region.instructions()[0], &supplied), Ok(None));
    }

    #[test]
    fn test_replay_type_identities_instruction_renaming() {
        // Reifying the same boundary axis twice can produce different first-class definitions. Each constructor's
        // renaming follows its own input definition, while the shared mapping for the array's boundary identity
        // stays unchanged.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(vector.clone().into());
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let size = builder
                .add_instruction(DimensionSizeOperation::new(&vector, 0).unwrap(), vec![], vec![input], None)
                .unwrap()[0];
            outputs.push(
                builder.add_instruction(ZeroOperation::new(vector.clone()), vec![], vec![size], None).unwrap()[0],
            );
        }
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let input_types = region.input_types();
        let refinements = ArrayIrTypeRefinements::new(&input_types, &input_types).unwrap();
        let replay = ReplayTypeIdentities::new(region, &input_types, refinements, true).unwrap();
        let first = DimensionType::new("first", n.bounds());
        let second = DimensionType::new("second", n.bounds());
        for (index, actual) in [(0, &first), (2, &second)] {
            replay
                .record_outputs(region, &region.instructions()[index], &[actual.clone().into()], None)
                .unwrap();
            let local = replay
                .instruction_renaming(region, &region.instructions()[index + 1], &[actual.clone().into()])
                .unwrap();
            assert_eq!(local.rename(n.variable().unwrap()), actual.variable().unwrap().clone());
        }
        assert!(replay.identity_renaming.borrow().is_identity());
        let local = replay.instruction_renaming(region, &region.instructions()[1], &[first.clone().into()]).unwrap();
        assert_eq!(local.rename(n.variable().unwrap()), first.variable().unwrap().clone());
    }

    #[test]
    fn test_replay_type_identities_record_outputs() {
        // Each replay records the fresh result of dimension arithmetic, and every later shape references that result.
        // Separate replay states must not share a substitution merely because they replay the same source region.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let addition = DimensionAddOperation::new(&n, &n).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(n.clone().into());
        let result = builder.add_instruction(addition, vec![], vec![input, input], None).unwrap()[0];
        let ArrayIrType::Dimension(result_type) = builder.atoms()[result.index()].r#type().into_owned() else {
            panic!("expected a dimension result");
        };
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![result_type.dimension().clone()]));
        let output = builder.add_instruction(ZeroOperation::new(vector), vec![], vec![result], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![result, output],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let actual_input = DimensionType::new("input", n.bounds());
        let input_types = vec![actual_input.clone().into()];
        let mut actual_results = Vec::new();
        for _ in 0..2 {
            let refinements = ArrayIrTypeRefinements::new(region.input_types(), &input_types).unwrap();
            let replay = ReplayTypeIdentities::new(region, &input_types, refinements, true).unwrap();
            let actual = DimensionType::new("result", result_type.bounds());
            replay.record_outputs(region, &region.instructions()[0], &[actual.clone().into()], None).unwrap();
            let mapping =
                replay.instruction_renaming(region, &region.instructions()[1], &[actual.clone().into()]).unwrap();
            assert_eq!(mapping.rename(n.variable().unwrap()), actual_input.variable().unwrap().clone());
            assert_eq!(mapping.rename(result_type.variable().unwrap()), actual.variable().unwrap().clone());
            let outputs = vec![
                actual.clone().into(),
                ArrayType::new(DataType::F32, Shape::new(vec![actual.dimension().clone()])).into(),
            ];
            let renamed_declared = region
                .output_types()
                .iter()
                .map(|r#type| r#type.rename_identities(&replay.identity_renaming.borrow()))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(renamed_declared, outputs);
            actual_results.push(actual);
        }
        assert_ne!(actual_results[0].variable(), actual_results[1].variable());

        // Ordinary replay must rename a definition to the replayed output's identity and rejects an output whose type
        // does not refine the declared result. Partial replay reports its input types, so an input that a proven fold
        // forwards unchanged (here `value`, wider than the source's capped `value + offset` result) introduces no
        // definition and keeps its own identity.
        let forwarded = DimensionType::new("value", DimensionBounds::unbounded());
        let offset = DimensionType::new("offset", DimensionBounds::new(0, Some(2)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(forwarded.clone().into());
        let offset_input = builder.add_input(offset.clone().into());
        let result = builder
            .add_instruction(
                DimensionAddOperation::new(&forwarded, &offset).unwrap(),
                vec![],
                vec![input, offset_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![result],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let forwarded_input_types =
            vec![forwarded.clone().into(), DimensionValue::constant(0).unwrap().r#type().into_owned().into()];
        let refinements = ArrayIrTypeRefinements::new(region.input_types(), &forwarded_input_types).unwrap();
        let replay = ReplayTypeIdentities::new(region, &forwarded_input_types, refinements, true).unwrap();
        assert_eq!(
            replay.record_outputs(region, &region.instructions()[0], &[forwarded.clone().into()], None),
            Err(ProgramError::Type(TypeError::invalid(
                "dimension type dimension<value ∈ [0, ∞)> cannot instantiate declared type \
                 dimension<value + offset ∈ [0, 9223372036854775808)>",
            ))),
        );
        replay
            .record_outputs(region, &region.instructions()[0], &[forwarded.into()], Some(&forwarded_input_types))
            .unwrap();
        assert!(replay.identity_renaming.borrow().is_identity());
    }

    #[test]
    fn test_replay_type_identities_rename_constant() {
        // Array IR literals are static and own no identities, so constant freshening is observable only on a type
        // whose constants define identities. A reference to such a constant follows the same substitution, while a
        // separate replay of the region gets a fresh owner.
        let identity = StructuralIdentity::new("constant");
        let definition = StructuralType::definition(identity.clone());
        let reference = StructuralType::reference(identity.clone());
        let mut builder = ProgramBuilder::<StructuralType, StructuralOperation>::new();
        let defined = builder.add_constant(definition.clone());
        let referenced = builder.add_constant(reference.clone());
        let program = builder
            .build::<Vec<StructuralType>, Vec<StructuralType>>(
                vec![defined, referenced],
                Vec::new(),
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        assert_eq!(region.type_identity_signature().internal_identities(), std::slice::from_ref(&identity));
        let first = ReplayTypeIdentities::new(region, &[], (), true).unwrap();
        let second = ReplayTypeIdentities::new(region, &[], (), true).unwrap();
        let first_definition = first.rename_constant(&definition).unwrap();
        let second_definition = second.rename_constant(&definition).unwrap();
        assert_ne!(first_definition.definitions, definition.definitions);
        assert_ne!(second_definition.definitions, definition.definitions);
        assert_ne!(first_definition.definitions, second_definition.definitions);
        let first_reference = first.rename_constant(&reference).unwrap();
        let second_reference = second.rename_constant(&reference).unwrap();
        assert_eq!(first_reference.references, first_definition.definitions);
        assert_eq!(second_reference.references, second_definition.definitions);

        // Reusing a replay's constant does not allocate a second identity or mutate the original payload.
        assert_eq!(first.rename_constant(&definition), Ok(first_definition.clone()));
        assert_eq!(definition.definitions, vec![identity]);
    }

    #[test]
    fn test_region_interpreter_new() {
        let program = stepping_program();
        let state = RegionInterpreter::new(
            TestArrayContext::new(),
            program.entry_region_ref(),
            vec![Array::scalar(2f32).unwrap()],
        )
        .unwrap();
        assert!(!state.is_complete());
        assert!(matches!(
            RegionInterpreter::new(TestArrayContext::new(), program.entry_region_ref(), vec![]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        let error = RegionInterpreter::new(
            TestArrayContext::new(),
            program.entry_region_ref(),
            vec![Array::scalar(2i32).unwrap()],
        );
        assert!(matches!(error, Err(error) if error == ProgramError::from(TypeError::invalid(
            "encountered input type i32[] which is incompatible with the region's declared type f32[]",
        ))));
    }

    #[test]
    fn test_region_interpreter_context() {
        let program = stepping_program();
        let context = TracingContext::<Array, TestArrayOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F32));
        let state = RegionInterpreter::new(context.clone(), program.entry_region_ref(), vec![input]).unwrap();
        assert!(std::rc::Rc::ptr_eq(state.context().builder(), context.builder()));
    }

    #[test]
    fn test_region_interpreter_is_complete() {
        let program = stepping_program();
        let mut state = RegionInterpreter::new(
            TestArrayContext::new(),
            program.entry_region_ref(),
            vec![Array::scalar(2f32).unwrap()],
        )
        .unwrap();
        assert!(!state.is_complete());
        state = state.step().unwrap();
        assert!(!state.is_complete());
        state = state.step().unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn test_region_interpreter_step() {
        let program = stepping_program();
        let context = TracingContext::<Array, TestArrayOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F32));
        let state = RegionInterpreter::new(context.clone(), program.entry_region_ref(), vec![input]).unwrap();
        let state = state.step().unwrap();
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        let state = state.step().unwrap();
        assert_eq!(context.builder().borrow().instructions().len(), 2);
        let outputs = state.finish().unwrap();
        assert_eq!(outputs[0].atom_id().unwrap(), outputs[1].atom_id().unwrap());
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::scalar(DataType::F32));
    }

    #[test]
    fn test_region_interpreter_finish() {
        let program = stepping_program();
        let state = RegionInterpreter::new(
            TestArrayContext::new(),
            program.entry_region_ref(),
            vec![Array::scalar(2f32).unwrap()],
        )
        .unwrap();
        assert_eq!(
            state.finish(),
            Err(ProgramError::MalformedProgram(
                "cannot finish region interpretation with 2 instructions remaining".to_owned(),
            )),
        );
        let state = RegionInterpreter::new(
            TestArrayContext::new(),
            program.entry_region_ref(),
            vec![Array::scalar(2f32).unwrap()],
        )
        .unwrap();
        assert_eq!(
            state.step().unwrap().step().unwrap().finish(),
            program
                .entry_region_ref()
                .interpret_in_context(&TestArrayContext::new(), vec![Array::scalar(2f32).unwrap()]),
        );
    }

    #[test]
    fn test_region_interpreter_state_new() {
        let program = stepping_program();
        let mut lifted = 0;
        let state =
            RegionInterpreterState::new(program.entry_region_ref(), vec![Array::scalar(2f32).unwrap()], |_, value| {
                lifted += 1;
                Ok::<_, ProgramError>(value.clone())
            })
            .unwrap();
        assert_eq!(lifted, 1);
        assert_eq!(state.next_instruction, 0);
        assert_eq!(state.remaining_uses[program.output_ids()[0].index()], 2);
        assert!(matches!(
            RegionInterpreterState::new(program.entry_region_ref(), Vec::<Array>::new(), |_, value| {
                Ok::<_, ProgramError>(value.clone())
            }),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_region_interpreter_state_is_complete() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let value = builder.add_constant(Array::scalar(3f32).unwrap());
        let program = builder.build::<Vec<Array>, Vec<Array>>(vec![value], vec![], vec![Placeholder]).unwrap();
        let state = RegionInterpreterState::new(program.entry_region_ref(), vec![], |_, value| {
            Ok::<_, ProgramError>(value.clone())
        })
        .unwrap();
        assert!(state.is_complete());
        assert_eq!(state.finish(), Ok(vec![Array::scalar(3f32).unwrap()]));
    }

    #[test]
    fn test_region_interpreter_state_step() {
        let program = stepping_program();
        let state =
            RegionInterpreterState::new(program.entry_region_ref(), vec![Array::scalar(2f32).unwrap()], |_, value| {
                Ok::<_, ProgramError>(value.clone())
            })
            .unwrap();
        let mut dispatched = Vec::new();
        let mut dispatch = |instruction: &Instruction<TestArrayOperation>, inputs: &[Array]| {
            dispatched.push(instruction.operation().name());
            instruction.operation().interpret(&TestArrayContext::new(), &EmptyRegionDriver, inputs)
        };
        let state = state.step(&mut dispatch).unwrap();
        assert!(!state.is_complete());
        let state = state.step(&mut dispatch).unwrap();
        assert!(state.is_complete());
        assert_eq!(dispatched, vec!["add", "neg"]);
        assert_eq!(state.finish(), Ok(vec![Array::scalar(-5f32).unwrap(), Array::scalar(-5f32).unwrap()]));
    }

    #[test]
    fn test_region_interpreter_state_step_rejects_wrong_output_count() {
        let program = stepping_program();
        let state =
            RegionInterpreterState::new(program.entry_region_ref(), vec![Array::scalar(2f32).unwrap()], |_, value| {
                Ok::<_, ProgramError>(value.clone())
            })
            .unwrap();
        assert!(matches!(
            state.step(|_, _| Ok::<Vec<Array>, ProgramError>(vec![])),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_region_interpreter_state_finish() {
        let program = stepping_program();
        let state =
            RegionInterpreterState::new(program.entry_region_ref(), vec![Array::scalar(2f32).unwrap()], |_, value| {
                Ok::<_, ProgramError>(value.clone())
            })
            .unwrap();
        assert_eq!(
            state.finish(),
            Err(ProgramError::MalformedProgram(
                "cannot finish region interpretation with 2 instructions remaining".to_owned(),
            )),
        );
    }

    #[test]
    fn test_interpret_projected_operation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let operation = ArrayOperation::Add(AddOperation::new());
        let inputs = [
            ArrayIrValue::Array(Array::vector(vec![1.0, 2.0]).unwrap()),
            ArrayIrValue::Array(Array::vector(vec![3.0, 4.0]).unwrap()),
        ];
        assert_eq!(
            interpret_projected_operation(&context, &operation, &EmptyRegionDriver, &inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0, 6.0]).unwrap())]),
        );
    }
}
