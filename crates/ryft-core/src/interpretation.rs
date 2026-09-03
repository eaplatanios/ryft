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
//! [`RegionRef`] provides corresponding context-driven and low-level methods that replay a borrowed sealed region
//! directly from its source arena.
//!
//! # Replay Environment and Boundaries
//!
//! [`Program::interpret_in_context`] first checks the input [`Parameterized`] structure and the complete input type
//! signature. Refinements are established across the whole signature so repeated dynamic type identities cannot receive
//! contradictory concrete bindings. Before replay, an eager concrete [`Value`] family may run its
//! [`Value::validate_eager_interpretation`] hook. For example, the core array IR uses this boundary to reject external
//! reference state before any mutation. Structured inputs are then flattened into an atom-indexed environment.
//! Only live constants are lifted through [`Context::lift`], and every instruction reads its operands and writes its
//! results in that environment.
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
//! Eager rules should request only the context capabilities they consume. Operand-driven rules often need only their
//! input values, while nullary constructors and captured constants may require focused capabilities such as zero
//! construction or constant materialization. The generic context parameter on [`InterpretableOperation`] is therefore
//! deliberately bounded by [`Domain`] rather than by [`Context`]. The trait's type-compatibility caveat is documented
//! on the trait itself.
//!
//! # Nested Regions
//!
//! Every [`Context::bind`] receives one instruction-scoped [`BindingRegionDriver`](crate::BindingRegionDriver). Eager
//! binding adapts that driver into an [`InterpretationDriver`] for the operation's eager rule. Region-free applications
//! use an empty driver. Higher-order eager rules select attached regions by index and recursively replay them through
//! the same active context. Borrowed regions remain in the source arena, and shared replay mappings preserve repeated
//! roots and descendants when another active context stages them into a destination arena. No standalone program is
//! materialized for recursive eager interpretation.
//!
//! # Composite and Projected Operations
//!
//! [`InterpretableOperation`] describes eager semantics in an operation's native value universe.
//! [`MemberInterpretableOperation`] handles a payload whose enclosing instruction consumes or produces several member
//! kinds. [`interpret_projected_operation`] adapts a region-free native member rule to a composite domain by projecting
//! operands, executing once through the member's eager context, and lifting results back into the composite value
//! family. Region-carrying and genuinely mixed operations remain composite-native because their region contracts may
//! span several member kinds.
//!
//! # Extending Interpretation
//!
//! Add eager semantics by implementing [`InterpretableOperation`] on operation payloads and dispatching
//! wrapper operation families to those implementations. Higher-order rules recurse through their supplied
//! [`InterpretationDriver`]. A new replay meaning normally requires no second interpreter: define a [`Context`]
//! whose [`Context::bind`] implements that meaning and call [`Program::interpret_in_context`].

use std::fmt::Debug;

use crate::contexts::{Context, Domain, EagerContext};
use crate::macros::check_count;
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::programs::{
    Atom, AtomId, EagerInterpretationValidation, EmptyRegionDriver, Instruction, Operation, Program, ProgramError,
    RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver, Type, TypeError, TypeRefinements, Typed, Value,
    ValueProjection,
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
}

/// Adapts the [`BindingRegionDriver`](crate::BindingRegionDriver) supplied to one eager [`Context::bind`] operation
/// application to the interpretation-specific recursion provided by [`InterpretationDriver`]. Its regions are exactly
/// the nested computations supplied to that application, and [`EagerInterpretationDriver::interpret_region`] replays a
/// selected region through the [`EagerContext`] without taking ownership of it.
pub(crate) struct EagerInterpretationDriver<'r, D> {
    /// Binding [`RegionDriver`] supplied to the active [`Operation`] application.
    driver: &'r D,

    /// Evidence that the complete attached region closure was validated before the active application's eager rule
    /// began executing, forwarded to nested replay so that selected regions are not revalidated.
    validation: Option<EagerInterpretationValidation>,
}

impl<'r, D> EagerInterpretationDriver<'r, D> {
    /// Creates a new [`EagerInterpretationDriver`].
    #[inline]
    pub(crate) fn new(driver: &'r D, validation: Option<EagerInterpretationValidation>) -> Self {
        Self { driver, validation }
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

impl<V: Value, O: Operation<Type = V::Type> + InterpretableOperation<EagerContext<V, O>>, D: RegionDriver<V, O>>
    InterpretationDriver<EagerContext<V, O>> for EagerInterpretationDriver<'_, D>
{
    #[inline]
    fn interpret_region(
        &self,
        context: &EagerContext<V, O>,
        index: usize,
        inputs: Vec<V>,
    ) -> Result<Vec<V>, ProgramError> {
        // `EagerContext::bind` ensures the complete attached region closure is validated before any eager rule runs,
        // so this recursion forwards that evidence instead of revalidating the selected child. Revalidation here
        // would happen after branch selection and would misclassify forwarded parent roots as external resources.
        self.region(index)?.interpret_in_context(context, inputs, self.validation.as_ref())
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
/// reaching [`Context`] through this trait would make that obligation recursive.
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
    ///   - `inputs`: Parent-universe inputs/operands in [`Instruction`] order.
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
        O: InterpretableOperation<EagerContext<V, O>>,
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
    /// interpretation (e.g., control flow branches, custom derivative programs, etc.) routes back through here via the
    /// driver behind the rule's [`InterpretationDriver`].
    pub fn interpret_in_context<C: Context<Type = T, Constant = V, Operation = O>>(
        &self,
        context: &C,
        input: Input::To<C::Value>,
    ) -> Result<Output::To<C::Value>, ProgramError>
    where
        O: Clone,
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

        // Flatten the structured input and validate the complete input type signature. Program construction normally
        // guarantees these identifiers, but every check below indexes the atom table directly. Validate the boundary
        // first so malformed internal state remains a structured error rather than a panic during diagnostic or
        // refinement handling.
        let inputs = input.into_parameters().collect::<Vec<_>>();
        let input_ids = self.input_ids();
        for input_id in input_ids {
            if input_id.index() >= self.atoms().len() {
                return Err(ProgramError::UnboundAtomId { id: *input_id });
            }
        }

        // Refinement errors that already carry structured cross-leaf or dimension details pass through unchanged.
        // For an ordinary pairwise incompatibility, retain the established program-boundary diagnostic naming the
        // first mismatched leaf and whether it occurred at the input or output boundary.
        let contextualize_refinement_error =
            |error: TypeError, ids: &[AtomId], actual: &[C::Value], position: &str| -> ProgramError {
                if matches!(&error, TypeError::Invalid { .. }) && ids.len() == actual.len() {
                    for (id, actual) in ids.iter().zip(actual) {
                        let declared = self.atoms()[id.index()].r#type();
                        let actual = actual.r#type();
                        if !declared.is_refined_by(actual.as_ref()) {
                            return TypeError::invalid(format!(
                                "encountered {position} type {actual} which is incompatible with the program's \
                                 declared type {declared}",
                            ))
                            .into();
                        }
                    }
                }
                error.into()
            };

        // Equal boundary types carry no additional refinement facts, so avoid constructing a refinement environment
        // in the common staging/replay case. More precise actual types (for example, a static extent supplied for a
        // dynamic dimension) must be checked as one complete signature so repeated identities agree across inputs.
        let refinements = if input_ids.len() == inputs.len()
            && input_ids
                .iter()
                .zip(&inputs)
                .all(|(id, actual)| self.atoms()[id.index()].r#type().as_ref() == actual.r#type().as_ref())
        {
            T::Refinements::default()
        } else {
            T::Refinements::establish(
                input_ids.iter().map(|id| self.atoms()[id.index()].r#type()),
                inputs.iter().map(Typed::r#type),
            )
            .map_err(|error| contextualize_refinement_error(error, input_ids, &inputs, "input"))?
        };

        // Concrete resource-bearing value families need a complete, all-or-nothing legality check before replay can
        // lift constants or execute the first instruction. Transform wrappers use their own value family and therefore
        // retain their operation-level gates even when their innermost execution context is eager. Runtime reference
        // alias validation is deliberately not performed here as it belongs to the public transform boundaries (e.g.,
        // `jvp`, `linearize`, `batch`, etc.) that bind concrete values, whereas this replay is also reached with
        // transform tracers over an eager parent, which report no identity, and with legitimately repeated residual
        // reference slots.
        let requires_validation = context.is_eager() && C::Value::VALIDATES_EAGER_INTERPRETATION;
        if requires_validation {
            C::Value::validate_eager_interpretation(self.entry_region_ref())?;
        }

        // Replay through the context's lift/bind protocol and reshape the flat outputs back into the expected
        // structured output form of this program, reparameterized at the context's value type. All instructions
        // share one mapping scope so that a staging destination imports each unchanged source region at most once.
        let source = self.entry_region_ref();
        let region_mappings = RegionReplayMappings::new();
        let outputs = self.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| {
                let driver = ReplayRegionDriver::with_validation(
                    source,
                    instruction.regions(),
                    &region_mappings,
                    requires_validation,
                )?;
                // Every replayed instruction binds inside its own recorded origin, so a one-to-one rewrite preserves
                // the source provenance exactly, and a one-to-many rewrite attaches it to every generated instruction.
                // This also holds for unknown source provenance: preserving it exactly means the replay must not absorb
                // ambient transform scopes.
                context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    context.bind(instruction.operation().clone(), driver, inputs)
                })
            },
        )?;

        // Replayed values commonly retain the program's exact declared output types. Only a count difference or a
        // more precise output type requires validation. That validation reuses facts established from the inputs and
        // permits a previously unbound output identity only when it belongs to this program's closed identity signature
        // (established by the formal inputs or defined by an instruction inside it).
        let output_ids = self.output_ids();
        if output_ids.len() != outputs.len()
            || output_ids
                .iter()
                .zip(&outputs)
                .any(|(id, actual)| self.atoms()[id.index()].r#type().as_ref() != actual.r#type().as_ref())
        {
            refinements
                .validate(
                    output_ids.iter().map(|id| self.atoms()[id.index()].r#type()),
                    outputs.iter().map(Typed::r#type),
                    self.type_identity_signature().identities(),
                )
                .map_err(|error| contextualize_refinement_error(error, output_ids, &outputs, "output"))?;
        }

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
    /// This low-level callback API does not run [`Value::validate_eager_interpretation`]. A caller whose `interpret_fn`
    /// can mutate resource-bearing values must validate the complete program and invocation before calling it.
    /// [`Program::interpret_in_context`] is the canonical checked eager entry point.
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
    ///   - `validation`: Evidence that a [`Value::validate_eager_interpretation`] boundary validation already covered
    ///     this region as part of an enclosing checked interpretation root. Pass [`None`] unless you hold evidence
    ///     obtained from such a root. [`None`] makes this interpretation its own root boundary, so that an eager
    ///     context whose value family validates eager interpretation runs the boundary validation here before
    ///     anything executes. Ryft's nested replay machinery forwards the evidence minted by its enclosing root
    ///     instead, both because that root's boundary validation already covered every nested region and because
    ///     revalidating a selected child in isolation would misclassify parent-created references forwarded into the
    ///     child as external roots.
    pub fn interpret_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: Vec<C::Value>,
        validation: Option<&EagerInterpretationValidation>,
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, self.input_ids().len(), ProgramError);
        let input_ids = self.input_ids();
        for input_id in input_ids {
            if input_id.index() >= self.atoms().len() {
                return Err(ProgramError::UnboundAtomId { id: *input_id });
            }
        }

        // Preserve structured refinement errors while keeping the established pairwise region-boundary diagnostic for
        // ordinary incompatibilities. This closure is local because the owner wording and atom lookup are properties
        // of this specific borrowed region, not a separate boundary-refinement abstraction.
        let contextualize_refinement_error =
            |error: TypeError, ids: &[AtomId], actual: &[C::Value], position: &str| -> ProgramError {
                if matches!(&error, TypeError::Invalid { .. }) && ids.len() == actual.len() {
                    for (id, actual) in ids.iter().zip(actual) {
                        let declared = self.atoms()[id.index()].r#type();
                        let actual = actual.r#type();
                        if !declared.is_refined_by(actual.as_ref()) {
                            return TypeError::invalid(format!(
                                "encountered {position} type {actual} which is incompatible with the region's declared \
                                 type {declared}",
                            ))
                            .into();
                        }
                    }
                }
                error.into()
            };

        // Exact input types introduce no new boundary facts. Otherwise, establish one environment across the entire
        // region signature so repeated dynamic identities cannot acquire conflicting concrete refinements.
        let refinements = if input_ids.len() == inputs.len()
            && input_ids
                .iter()
                .zip(&inputs)
                .all(|(id, actual)| self.atoms()[id.index()].r#type().as_ref() == actual.r#type().as_ref())
        {
            <V::Type as Type>::Refinements::default()
        } else {
            <V::Type as Type>::Refinements::establish(
                input_ids.iter().map(|id| self.atoms()[id.index()].r#type()),
                inputs.iter().map(Typed::r#type),
            )
            .map_err(|error| contextualize_refinement_error(error, input_ids, &inputs, "input"))?
        };

        // A replay without validation evidence is a root execution boundary just like `Program::interpret_in_context`.
        // Nested eager recursion supplies its root's evidence instead, because that root's boundary validation
        // already covered the whole closure in its original context.
        let validated = match validation {
            Some(_) => true,
            None => {
                let requires_boundary_validation = context.is_eager() && C::Value::VALIDATES_EAGER_INTERPRETATION;
                if requires_boundary_validation {
                    C::Value::validate_eager_interpretation(self)?;
                }
                requires_boundary_validation
            }
        };

        // Share one source-to-destination mapping across every instruction in this replay. If several instructions
        // attach the same nested source region, a staging context imports it only once and preserves that sharing.
        let region_mappings = RegionReplayMappings::new();
        let outputs = self.interpret_with(
            inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| {
                let driver =
                    ReplayRegionDriver::with_validation(self, instruction.regions(), &region_mappings, validated)?;
                // Refer to the matching comment in `Program::interpret_in_context`. Binding inside the source
                // instruction's recorded origin makes one-to-one and one-to-many propagation automatic.
                context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    context.bind(instruction.operation().clone(), driver, inputs)
                })
            },
        )?;

        // Skip output validation when replay preserved every declared type exactly. Refined outputs are checked
        // against the input environment. Only identities in this region's closed identity signature (established by
        // its formal inputs or defined by its instructions) may establish new facts at the output boundary.
        let output_ids = self.output_ids();
        if output_ids.len() != outputs.len()
            || output_ids
                .iter()
                .zip(&outputs)
                .any(|(id, actual)| self.atoms()[id.index()].r#type().as_ref() != actual.r#type().as_ref())
        {
            refinements
                .validate(
                    output_ids.iter().map(|id| self.atoms()[id.index()].r#type()),
                    outputs.iter().map(Typed::r#type),
                    self.type_identity_signature().identities(),
                )
                .map_err(|error| contextualize_refinement_error(error, output_ids, &outputs, "output"))?;
        }

        Ok(outputs)
    }

    /// Interprets/executes this borrowed [`RegionRef`]'s [`Instruction`]s using the caller-supplied value and error
    /// semantics. This is the borrowed-[`Region`](crate::Region) counterpart of [`Program::interpret_with`]. It replays
    /// the region directly from its source arena without first materializing a standalone [`Program`], while preserving
    /// the same flat input, constant-lifting, instruction-dispatch, and output-gathering behavior.
    ///
    /// This low-level callback API does not run [`Value::validate_eager_interpretation`]. A caller whose `interpret_fn`
    /// can mutate resource-bearing values must validate the complete reachable region closure and its invocation before
    /// calling it. [`RegionRef::interpret_in_context`] is the canonical checked eager entry point.
    pub fn interpret_with<
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftFn: FnMut(AtomId, &V) -> Result<RuntimeValue, Error>,
        InterpretFn: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        self,
        inputs: Vec<RuntimeValue>,
        mut lift_fn: LiftFn,
        mut interpret_fn: InterpretFn,
    ) -> Result<Vec<RuntimeValue>, Error> {
        let atoms = self.atoms();
        let input_ids = self.input_ids();
        let instructions = self.instructions();
        let output_ids = self.output_ids();
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

        // Materialize literal constants that are live. Dead constants can remain unset because no instruction or
        // region output will read them.
        for (atom_index, atom) in atoms.iter().enumerate() {
            if remaining_uses[atom_index] == 0 {
                continue;
            }
            if let Atom::Constant(value) = atom {
                values[atom_index] = Some(lift_fn(AtomId::new(atom_index), value)?);
            }
        }

        // Replay instructions in region order, reusing one scratch input buffer to avoid per-instruction allocation.
        let max_input_count = instructions.iter().map(|instruction| instruction.inputs().len()).max().unwrap_or(0);
        let mut instruction_inputs = Vec::with_capacity(max_input_count);
        for instruction in instructions {
            instruction_inputs.clear();
            for input_id in instruction.inputs().iter().copied() {
                // Consume the appropriate input value for the current instruction. If this is the last consumer,
                // move the value out of the table. Otherwise, clone it so later consumers can still read it.
                let remaining_uses = remaining_uses.get_mut(input_id.index()).unwrap();
                debug_assert!(*remaining_uses > 0);
                *remaining_uses -= 1;
                let value = values.get_mut(input_id.index()).unwrap();
                let value = if *remaining_uses == 0 { value.take().unwrap() } else { value.as_ref().unwrap().clone() };
                instruction_inputs.push(value);
            }

            // Apply the operation using the supplied dispatcher and ensure it produces the expected number of outputs.
            let outputs = interpret_fn(instruction, instruction_inputs.as_slice())?;
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);

            for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
                let Some(value) = values.get_mut(output_id.index()) else {
                    return Err(ProgramError::UnboundAtomId { id: output_id }.into());
                };

                // Keep only outputs with a future consumer. Dead instruction results do not need to occupy the table.
                if remaining_uses[output_id.index()] != 0 {
                    *value = Some(output);
                }
            }
        }

        // Gather the region outputs using the same last-use transfer logic that we used for instruction inputs.
        let mut outputs = Vec::with_capacity(output_ids.len());
        for output_id in output_ids.iter().copied() {
            let remaining_uses = remaining_uses.get_mut(output_id.index()).unwrap();
            debug_assert!(*remaining_uses > 0);
            *remaining_uses -= 1;
            let value = values.get_mut(output_id.index()).unwrap();
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
    use std::cell::Cell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds,
        DimensionError, DimensionVariable, Shape,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{AddOperation, BroadcastOperation, NegOperation};
    use crate::parameters::{ParameterError, Parameterized, Placeholder};
    use crate::programs::{
        AtomId, BindingRegionDriver, ProgramBuilder, ProgramError, Provenance, ProvenanceScope, RegionInterface,
        TypeError,
    };
    use crate::tests::TestRegionOperation;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_empty_region_driver_interpret_region() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let expected = ProgramError::MalformedProgram("empty region driver cannot interpret a region".to_string());
        let expected_region = ProgramError::MalformedProgram("region index 0 is out of range".to_string());
        assert_eq!(RegionDriver::<Array, ArrayOperation<Array>>::regions(&EmptyRegionDriver).count(), 0);
        assert_eq!(RegionDriver::<Array, ArrayOperation<Array>>::region_count(&EmptyRegionDriver), 0);
        assert!(matches!(
            RegionDriver::<Array, ArrayOperation<Array>>::region(&EmptyRegionDriver, 0),
            Err(error) if error == expected_region,
        ));
        assert_eq!(EmptyRegionDriver.interpret_region(&context, 0, Vec::<Array>::new()), Err(expected));
    }

    #[test]
    fn test_program_interpret_materializes_duplicate_outputs() {
        // A program whose two outputs are the same atom materializes that value into both output positions.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F32));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder
            .build::<Array, (Array, Array)>(vec![o0, o0], Placeholder, (Placeholder, Placeholder))
            .unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0f32)), Ok((Array::scalar(4.0f32), Array::scalar(4.0f32))));
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
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![shared],
                vec![source_input],
                None,
            )
            .unwrap()[0];
        let second = source_builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
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
        assert_eq!(destination.instructions()[0].regions(), &[crate::RegionId::new(0)]);
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
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
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
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
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
    fn test_program_interpret_marks_replay_validated_only_after_running_the_value_hook() {
        /// Eager test context that records whether program replay supplied privileged boundary-validation evidence.
        #[derive(Clone)]
        struct ReplayEvidenceContext {
            /// Whether the observed application used an ordinary unvalidated replay driver.
            saw_unvalidated_replay: Rc<Cell<bool>>,
        }

        impl Domain for ReplayEvidenceContext {
            type Type = ArrayType;
            type Value = Array;
            type Constant = Array;
            type Operation = TestRegionOperation;
        }

        impl Context for ReplayEvidenceContext {
            fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
                Ok(constant)
            }

            fn bind<O: Into<TestRegionOperation>, D: BindingRegionDriver<Array, TestRegionOperation>>(
                &self,
                operation: O,
                driver: D,
                inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                let operation = operation.into();
                operation.validate_region_count(driver.region_count())?;
                self.saw_unvalidated_replay.set(driver.eager_interpretation_validation().is_none());
                Ok(inputs.to_vec())
            }

            fn is_eager(&self) -> bool {
                true
            }

            fn provenance(&self) -> Provenance {
                // This test context executes eagerly and records no instructions, so provenance is a no-op.
                Provenance::unknown()
            }

            fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
                function()
            }

            fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
                function()
            }
        }

        let mut nested_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested_input = nested_builder.add_input(ArrayType::scalar(DataType::F64));
        let nested = nested_builder
            .build::<Vec<Array>, Vec<Array>>(vec![nested_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let nested = builder.import_program(nested);
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![nested],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let saw_unvalidated_replay = Rc::new(Cell::new(false));
        let context = ReplayEvidenceContext { saw_unvalidated_replay: Rc::clone(&saw_unvalidated_replay) };

        assert_eq!(program.interpret_in_context(&context, vec![Array::scalar(1.0_f64)]), Ok(vec![Array::scalar(1.0)]));
        assert!(saw_unvalidated_replay.get());

        saw_unvalidated_replay.set(false);
        assert_eq!(
            program.entry_region_ref().interpret_in_context(&context, vec![Array::scalar(1.0_f64)], None),
            Ok(vec![Array::scalar(1.0)]),
        );
        assert!(saw_unvalidated_replay.get());
    }

    #[test]
    fn test_program_interpret_lifts_live_constants_once() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let c0 = builder.add_constant(Array::scalar(7.0f64));
        let c1 = builder.add_constant(Array::scalar(3.0f64));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, c1], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        let mut lifted_constants = Vec::new();
        assert_eq!(
            program.interpret_with(
                vec![Array::scalar(2.0f64)],
                |atom_id, value| {
                    lifted_constants.push((atom_id, value.clone()));
                    Ok(value.clone())
                },
                |instruction, inputs| instruction.operation().interpret(
                    &EagerContext::<Array, ArrayOperation<Array>>::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Ok(vec![Array::scalar(5.0f64)]),
        );
        assert_eq!(lifted_constants, vec![(c1, Array::scalar(3.0f64))]);
        assert_eq!(c0, AtomId::new(1));
    }

    #[test]
    fn test_program_interpret_with_mismatched_parameter_structures() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Vec<Array>, Array>(vec![i0], vec![Placeholder], Placeholder).unwrap();
        assert!(matches!(
            program.interpret(vec![Array::scalar(1.0f64), Array::scalar(2.0f64)]),
            Err(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![Placeholder])
                && right_structure
                    == format!("{:?}", vec![Array::scalar(1.0f64), Array::scalar(2.0f64)].parameter_structure())
        ));
    }

    #[test]
    fn test_program_interpret_input_type_checking() {
        // A statically typed program input rejects values whose concrete types do not match it exactly.
        let mut builder = ProgramBuilder::<Array, AddOperation<ArrayType>>::new();
        let i0 = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0])).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(Array::vector(vec![1.0, 2.0, 3.0])),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message
                == "encountered input type f64[3] which is incompatible with the program's declared type f64[2]",
        ));

        // An unbounded dynamically sized program input accepts concrete values of any size, so one staged program
        // replays at several concrete sizes. Rank mismatches are still rejected.
        let mut builder = ProgramBuilder::<Array, AddOperation<ArrayType>>::new();
        let i0 = builder.add_input(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        ));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0])).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0, 3.0])).unwrap().to_f64s(), vec![2.0, 4.0, 6.0]);
        assert!(matches!(
            program.interpret(Array::scalar(1.0)),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message
                == "encountered input type f64[] which is incompatible with the program's declared type f64[dynamic]",
        ));

        // A bounded dynamically sized program input enforces its exclusive upper bound on concrete sizes.
        let mut builder = ProgramBuilder::<Array, AddOperation<ArrayType>>::new();
        let i0 = builder.add_input(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "dynamic",
                DimensionBounds::non_negative(Some(3)).unwrap(),
            ))]),
        ));
        let o0 = builder.add_instruction(AddOperation::new(), Vec::new(), vec![i0, i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::vector(vec![1.0, 2.0])).unwrap().to_f64s(), vec![2.0, 4.0]);
        assert!(matches!(
            program.interpret(Array::vector(vec![1.0, 2.0, 3.0])),
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
    fn test_program_interpret_boundary_refinements() {
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

        impl InterpretableOperation<EagerContext<Array, WrongShapeOperation>> for WrongShapeOperation {
            fn interpret<D: InterpretationDriver<EagerContext<Array, WrongShapeOperation>>>(
                &self,
                _context: &EagerContext<Array, WrongShapeOperation>,
                _driver: &D,
                _inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                Ok(vec![Array::vector(vec![1.0, 2.0, 3.0])])
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
            program.interpret((Array::vector(vec![1.0, 2.0]), Array::vector(vec![1.0, 2.0, 3.0]))),
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
            program.interpret(Array::vector(vec![1.0, 2.0])),
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
                    vec![Array::vector(vec![1.0, 2.0])],
                    None,
                ),
            Err(ProgramError::Type(error))
                if error.downcast_custom::<DimensionError>()
                    == Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 2,
                        actual: 3,
                    }),
        ));
    }

    #[test]
    fn test_program_interpret_does_not_refine_operation_payloads() {
        // Batching a program with a dynamic per-item dimension stages a homogeneous `broadcast` whose stored output
        // type retains that dynamic extent (refer to `test_program_batched_carries_dynamic_per_item_dimensions`).
        // The same payload is built directly here so that the limitation is pinned independently of batching.
        let dynamic = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()));
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic.clone()]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), dynamic]));
        let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
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

        // This pins an accepted limitation. Replay refines only program and region boundaries, never the types stored
        // inside an `Instruction`'s operation payload, so the boundary identity `n` is refined to `3` while the payload
        // keeps naming `n`. The payload therefore re-runs its own inference against the refined input and fails with
        // its own inference diagnostic, phrased purely in terms of the broadcasting rule and not in terms of the
        // refinement the boundary established. Refining the payload would require substituting extents into stored
        // types, which the type model deliberately does not provide: extents reach an operation only as explicit
        // Static Single Assignment (SSA) operands, as in `DynamicBroadcastOperation`. A diagnostic that named the
        // stale payload would have to originate from a structured operation error that causally identifies the failing
        // payload constraint, because the surfaced error alone does not establish that cause.
        let error = program.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0])]).unwrap_err();
        assert!(matches!(error, ProgramError::Type(_)));
        assert_eq!(
            error.to_string(),
            "broadcasting input axis 0 has size 3 but the output has size n; a dynamic dimension only broadcasts to \
             an identical dynamic dimension",
        );

        // A statically typed payload establishes no boundary facts, so ordinary replay is untouched.
        let mut builder = ProgramBuilder::<Array, BroadcastOperation>::new();
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
            program.interpret(vec![Array::vector(vec![1.0, 2.0, 3.0])]),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])]),
        );
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_inputs() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Array, Array>(vec![i0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                Vec::<Array>::new(),
                |_, value| Ok(value.clone()),
                |instruction, inputs| instruction.operation().interpret(
                    &EagerContext::<Array, ArrayOperation<Array>>::new(),
                    &EmptyRegionDriver,
                    inputs,
                ),
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_program_interpret_with_wrong_number_of_operation_outputs() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i0 = builder.add_input(ArrayType::scalar(DataType::F64));
        let o0 = builder.add_instruction(NegOperation::new(), Vec::new(), vec![i0], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![o0], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.interpret_with(
                vec![Array::scalar(2.0f64)],
                |_, value| Ok(value.clone()),
                |_, _| Ok::<Vec<Array>, ProgramError>(Vec::new()),
            ),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        ));
    }

    #[test]
    fn test_interpret_projected_operation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let operation = ArrayOperation::Add(AddOperation::new());
        let inputs =
            [ArrayIrValue::Array(Array::vector(vec![1.0, 2.0])), ArrayIrValue::Array(Array::vector(vec![3.0, 4.0]))];
        assert_eq!(
            interpret_projected_operation(&context, &operation, &EmptyRegionDriver, &inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0, 6.0]))]),
        );
    }
}
