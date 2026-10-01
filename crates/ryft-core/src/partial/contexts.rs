use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::contexts::{Context, Domain, ValueResolution};
use crate::parameters::{Parameter, Placeholder};
use crate::partial::evaluations::PartialEvaluation;
use crate::partial::operations::{PartiallyEvaluatableOperation, RecursivePartialEvaluationDriver};
use crate::partial::partitions::{EffectOrdering, PartitionedProgram};
use crate::partial::residuals::{ResidualPlacement, ResidualPolicyReference};
use crate::partial::values::{
    PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartialValueMaterialization,
};
use crate::programs::operations::OperationFoldReplacement;
use crate::programs::{
    AtomId, BindingRegionDriver, EffectClasses, FlatProgram, InstructionId, Operation, OperationPayloadProjection,
    Program, ProgramBuilder, ProgramError, ProjectedValue, Provenance, ProvenanceScope, ProvenanceState,
    ReferenceAnalysis, ReferenceIdentity, RegionRef, RegionReplayMappings, RegionRole, ReplayRegionDriver, Type,
    TypeError, TypeIdentityPosition, Typed, Value, ValueProjection,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::contexts::StagingContext;

/// Placement of _known_ reference operations when the known-side [`Context`] of a [`PartialEvaluationContext`] is
/// eager. Under a [`StagingContext`] known side the two placements coincide, because folding stages an operation into
/// the outer program instead of executing it, and both follow the ordering rules on
/// [`PartialEvaluationContext`]: once an ordered effect is deferred, later ordered effects remain residual too.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReferencePlacement {
    /// Known reference operations fold like every other ordered effect. Under an eager known side they execute at
    /// partial evaluation time unless an earlier ordered effect has been deferred. This is the placement of
    /// linearization, whose known side _is_ the evaluation of the primal program at the linearization point, so the
    /// primal accesses must run (and mutate the primal references) exactly as forward mode would, while the tangent
    /// accesses, which consume unknown tangent references, stage into the linear program.
    Execute,

    /// Under an eager known side, every reference operation (i.e., an operation with a reference-typed input or output,
    /// or a region-carrying operation whose closure touches references) stages into the residual program, and the live
    /// handles it touches are passed as known reference inputs (i.e., [`PartialEvaluation::known_reference_inputs`]),
    /// so partial evaluation never reads, mutates, or allocates live state and every access runs once per residual
    /// program run, observing the state of that run. This is the placement of specialization (i.e.,
    /// [`Program::partially_evaluate`] and the other program-replay entry points): the residual program is a reusable
    /// program that runs at some later time, possibly repeatedly and after the state has changed. An eager known side
    /// also has no way to prove that an allocation's complete lifecycle stays internal to the folded prefix, which is
    /// why allocations stage too. Under a staging known side this placement is inert, exactly like
    /// [`Execute`](Self::Execute): a known reference operation folds into the outer program and runs once per execution
    /// of that program, ahead of the residual program, unless an earlier ordered operation has residualized. The
    /// per-run guarantee therefore holds for the residual program relative to the outer program that consumes it,
    /// and not for the folded known accesses on their own.
    Stage,
}

/// Active [`Context`] that folds known work through a parent context `C` and records unknown-dependent work in a
/// residual [`ProgramBuilder`]. It drives [`Program::partially_evaluate`], [`Program::partially_evaluate_in_context`],
/// and transform interpreters that bind operations directly over [`PartialTracer`]s.
///
/// Each [`Context::bind`] dispatches the operation's [`PartiallyEvaluatableOperation::partially_evaluate`]
/// implementation. The default [`fold_or_residualize`](Self::fold_or_residualize) policy first applies supported
/// [`Operation::fold`] substitutions, then binds eligible all-known operations through the parent context or emits
/// residual work. Specialized rules can instead inline nested programs or preserve more known work.
///
/// # Evaluation Pipeline
///
/// ```mermaid
/// flowchart TD
///   inputs["Inputs Classified as Known Values or Unknown Types"] --> context["PartialEvaluationContext"]
///   context --> rules["Operation Partial-Evaluation Rules"]
///   rules -->|"all inputs known"| parent["Known-Side Parent Context"]
///   parent -->|"execute eagerly or append to an outer program"| known["Known Result Values"]
///   rules -->|"mixed or unknown"| residualize["Residualize Operation"]
///   unknown["Unknown Inputs and Residual Variables"] --> residualize
///   known -->|"needed by residual work"| materialize["Materialization Policy"]
///   materialize -->|"known variable"| residual_input["Residual Input Feeder"]
///   materialize -->|"literal or designated constant"| residual_constant["Inline Residual Constant"]
///   residual_input --> builder["Residual Program Builder"]
///   residual_constant --> builder
///   residualize --> builder
///   known --> finalize["Finalize Boundary Mappings"]
///   builder --> finalize
///   finalize --> result["Residual Program with Input and Output Wiring"]
/// ```
///
/// # Preserving Effect Order
///
/// The split executes all known work before residual work. Once any ordered operation residualizes, every later
/// ordered operation residualizes too, irrespective of its input knowledge, reference root, or effect class. This
/// preserves the order in which reference accesses can fail or synchronize relative to later mutations, assertions,
/// and I/O. Pure operations keep folding whenever their operands permit.
///
///   | Work                                    | Default Placement                                     |
///   | --------------------------------------- | ----------------------------------------------------- |
///   | Pure operation with known operands      | Parent context                                        |
///   | Operation with an unknown operand       | Residual program                                      |
///   | Ordered effect with none yet deferred   | Parent context when its operands are known            |
///   | Ordered effect after one is deferred    | Residual program, including effects on distinct roots |
///   | Any effect in a deferred sibling        | Residual program, even with known operands            |
///   | Eager reference work with `Stage`       | Residual program, including allocation and accesses   |
///
/// [`deferred_sibling`](Self::deferred_sibling) retains every effect in residual execution, including allocations with
/// known initializers. Pure known computations still fold through the parent. Separately invoked programs require an
/// explicit source-level partition that validates their dependencies and allocation lifetimes; changing a context does
/// not implicitly establish independence between reference roots.
///
/// Control flow rules preserve the same contract. A known `condition` inlines only its selected branch under the
/// enclosing context's ordering constraints, and an unknown ordered `condition` residualizes whole. Ordinary `scan`
/// partitioning keeps a body whole when ordered effects would run on both sides. One-sided effects retain their
/// iteration order. Ordered `while` operations stay whole. Dormant derivative regions contribute neither effects nor
/// reference accesses until a transform invokes them. Speculative probes never execute effects through the live parent
/// or change its recorded ordering constraints; structural discovery uses fresh staging builders.
///
/// Mutable state is shared behind `Rc<RefCell<...>>` handles. Cloning this context therefore keeps every clone writing
/// to the same residual program, input descriptors, staged-feeder table, and ordering state, and lets rules re-enter
/// the context (e.g., to inline a selected condition branch through [`inline_program`](Self::inline_program)).
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct PartialEvaluationContext<C: Context> {
    /// Parent context in which known operations are evaluated or staged. An `Rc` shares the exact parent object across
    /// clones and deferred siblings instead of cloning `C`, whose `Clone` implementation need not preserve object
    /// identity. Imports compare these shared allocations to verify that a value comes from the same parent, even
    /// when the parent cannot resolve the value's identity. Independently constructed contexts have distinct parent
    /// allocations, while deferred siblings share the parent but build separate residual programs.
    parent: Rc<C>,

    /// [`ProgramBuilder`] accumulating the residual program's [`Atom`](crate::Atom)s and
    /// [`Instruction`](crate::Instruction)s. Shared across clones of this context (and held behind a [`RefCell`] for
    /// interior mutability) for the same reason that [`TracingContext`] shares its builder (i.e., values stamped with
    /// cloned contexts must keep accumulating into the _same_ residual program).
    builder: Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>>,

    /// Describes where each residual program input comes from, in input order, and is shared alongside the builder.
    inputs: Rc<RefCell<Vec<PartialEvaluationInput<C::Value>>>>,

    /// Map that is used for deduplication of residual _input_ feeders by the known value's _staged_ identity
    /// (i.e., the outer program [`Atom`](crate::Atom) a known value names when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in a staging known-side context), mapping the staged atom to the
    /// residual input already created for it. This complements the per-value shared [`PartialValueMaterialization`]
    /// slots along the axis that value-identity deduplication cannot reach: two _distinct_ known values (with distinct
    /// slots) naming the same outer atom collapse to one residual input, even when rule-produced. It holds only under a
    /// _staging_ known-side context because an eager context resolves knowns as [`Constant`](ValueResolution::Constant)
    /// rather than [`Staged`](ValueResolution::Staged), and so nothing is ever recorded in that case, and inline
    /// constants are excluded because they carry no staged identity.
    staged_feeders: Rc<RefCell<HashMap<AtomId, AtomId>>>,

    /// Cache containing known values imported from sibling contexts keyed by the address of their source
    /// materialization slot. Each import needs a slot in this context because source atom identifiers belong to a
    /// different residual builder; repeated imports reuse that slot. Retaining the source slot prevents its address
    /// from being reused for another value while the cache entry exists. Clones share the cache alongside the builder.
    imported_known_values:
        Rc<RefCell<HashMap<usize, (Rc<Cell<PartialValueMaterialization>>, PartialEvaluationValue<C::Value>)>>>,

    /// Maps materialized reference constants to their identities in the parent context. Residual uses of those
    /// constants must retain the same allocation identity, but constants have no entry in the input descriptors from
    /// which to recover it. A stored [`None`] value records that the parent identity is unresolved, preventing fallback
    /// to a misleading identity in the residual builder. Clones share this map alongside the builder.
    constant_reference_identities: Rc<RefCell<HashMap<AtomId, Option<ReferenceIdentity>>>>,

    /// Active [`ProvenanceState`] that residual [`Instruction`](crate::Instruction)s snapshot.
    /// [`PartialEvaluationContext`]s are staging boundaries (i.e., they own the residual [`ProgramBuilder`] and emit
    /// into it directly), and so they own provenance state exactly like tracing contexts instead of delegating reads
    /// to their known-side parents, which are often terminal eager contexts that would erase source provenance from
    /// every residual program. The state is seeded from the parent context's current provenance at construction and
    /// shared across clones.
    provenance: Rc<ProvenanceState>,

    /// Specifies whether known reference operations execute or remain residual when the parent is eager.
    /// Specialization retains these operations so they observe state when the specialized program is called.
    reference_placement: ReferencePlacement,

    /// Specifies whether effectful operations with known inputs may run in the parent context. Deferred sibling
    /// contexts disable this so that every residual call performs its own effects, including allocations with known
    /// initial values. Pure known work can still fold; reference placement and previously deferred ordered effects
    /// can further restrict folding.
    allow_effect_folding: bool,

    /// Specifies whether ordered effects must remain residual to preserve execution order. Clones share this flag so
    /// that nested operations cannot move effects ahead of already deferred work. During replay with per-reference
    /// ordering, each instruction receives a separate flag initialized from its conflicts with earlier deferred work;
    /// the reference sets themselves remain local to that replay.
    defer_ordered_effects: Rc<Cell<bool>>,

    /// Residual placement with which the partitions that split rules construct through
    /// [`PartialEvaluationDriver::partition_program`](crate::PartialEvaluationDriver::partition_program) place their
    /// residuals, if a residual policy was set. Refer to [`with_residual_policy`](Self::with_residual_policy) for how
    /// to do that.
    residual_placement: Option<Rc<dyn ResidualPlacement<C::Constant, C::Operation>>>,

    /// First binding error retained for finalization, even if the failed operation has no outputs or its poisoned
    /// outputs are discarded. Once set, later binds propagate poison without executing operations. Clones share the
    /// error so a failure in nested work cannot be lost when returning to the enclosing computation.
    error: Rc<RefCell<Option<ProgramError>>>,
}

impl<C: Context> PartialEvaluationContext<C> {
    /// Creates a fresh [`PartialEvaluationContext`] that folds known work through `parent` and accumulates residual
    /// work in a new residual [`ProgramBuilder`], with the [`Execute`](ReferencePlacement::Execute) placement of known
    /// reference operations. Refer to [`Self::with_reference_placement`] for the specialization placement.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::from_shared_parent(Rc::new(parent))
    }

    /// Creates fresh residual state around a shared parent, retaining its identity for known-value imports.
    fn from_shared_parent(parent: Rc<C>) -> Self {
        let provenance = Rc::new(ProvenanceState::seeded(parent.provenance()));
        Self {
            parent,
            builder: Rc::new(RefCell::new(ProgramBuilder::new())),
            inputs: Rc::new(RefCell::new(Vec::new())),
            staged_feeders: Rc::new(RefCell::new(HashMap::new())),
            imported_known_values: Rc::new(RefCell::new(HashMap::new())),
            constant_reference_identities: Rc::new(RefCell::new(HashMap::new())),
            provenance,
            reference_placement: ReferencePlacement::Execute,
            allow_effect_folding: true,
            defer_ordered_effects: Rc::new(Cell::new(false)),
            residual_placement: None,
            error: Rc::new(RefCell::new(None)),
        }
    }

    /// Returns this context with the provided [`ReferencePlacement`] for known reference operations under an eager
    /// parent. For example, use [`Stage`](ReferencePlacement::Stage) for specialization so reference operations run
    /// when the residual program is called rather than while it is constructed. Under a staging parent, both placements
    /// record known work in the parent program, subject to effect-ordering constraints.
    ///
    /// Note that this function changes the configuration of the returned context without creating a new residual
    /// program or changing previously emitted work. Existing clones retain their own placement setting.
    #[inline]
    pub fn with_reference_placement(mut self, reference_placement: ReferencePlacement) -> Self {
        self.reference_placement = reference_placement;
        self
    }

    /// Returns this context with the provided permission to fold effectful operations with known inputs into the parent
    /// context. When `false`, effectful operations remain in the residual program while pure known work can still fold.
    /// When `true`, reference placement and effect-ordering constraints can still prevent folding.
    ///
    /// This function changes only the returned context's configuration, without creating a new residual program or
    /// changing previously emitted work. Existing clones retain their own effect-folding setting.
    #[inline]
    pub fn with_allow_effect_folding(mut self, allow_effect_folding: bool) -> Self {
        self.allow_effect_folding = allow_effect_folding;
        self
    }

    /// Returns this context configured to place the residuals of its nested partitions according to `policy`.
    /// The split rules of region-carrying operations (e.g., `scan` and `condition`) partition their bodies through
    /// [`PartialEvaluationDriver::partition_program`](crate::PartialEvaluationDriver::partition_program), which then
    /// places the residuals of each such partition according to `policy` (which can be set using
    /// [`PartitionedProgram::with_residual_policy`]) and carries `policy` into the partitions nested within it. This
    /// has implications for various region-carrying operation types. For example, it means that decisions apply per
    /// iteration of a `scan` operation and per branch of a `condition` operation. When the residuals of an enclosing
    /// partition are placed, a region-carrying operation in its known program is replayed as a whole only if the policy
    /// would recompute every value that its regions compute, so that the decisions of a split rule that already placed
    /// the residuals of its body are never undone, while an operation whose inputs are all known (and which was
    /// therefore not split) is recomputed whenever the policy recomputes everything that it computes. A `while` loop
    /// is an exception that saves nothing, because its residual loop re-runs every iteration, and so its split rule
    /// partitions its regions without the policy.
    ///
    /// This function changes only the returned context's configuration, without creating a new residual program
    /// or changing previously emitted work. Existing clones retain their own residual policy.
    #[inline]
    pub fn with_residual_policy(self, policy: &ResidualPolicyReference<C::Type>) -> Self
    where
        C::Type: 'static,
        C::Operation: OperationPayloadProjection,
    {
        self.with_residual_placement(Some(Rc::new(policy.clone())))
    }

    /// Returns this context without a residual policy, for split rules whose residual programs do not consume
    /// saved values (e.g., a `while` loop, whose residual loop re-runs every iteration). Refer to
    /// [`with_residual_policy`](Self::with_residual_policy) for the inverse operation.
    #[inline]
    pub fn without_residual_policy(self) -> Self {
        self.with_residual_placement(None)
    }

    /// Sets whether ordered effects must remain residual, using a new flag that is independent
    /// of existing context clones. Future clones of the returned context share the new flag.
    /// Refer to [`defer_ordered_effects`](Self::defer_ordered_effects) for more information.
    ///
    /// Per-reference replay uses this to initialize each instruction's deferral flag according to whether its effects
    /// conflict with earlier deferred work.
    #[inline]
    pub(super) fn with_defer_ordered_effects(mut self, defer_ordered_effects: bool) -> Self {
        self.defer_ordered_effects = Rc::new(Cell::new(defer_ordered_effects));
        self
    }

    /// Returns this context with the provided residual placement, which nested contexts inherit
    /// from their parents.
    #[inline]
    pub(super) fn with_residual_placement(
        mut self,
        residual_placement: Option<Rc<dyn ResidualPlacement<C::Constant, C::Operation>>>,
    ) -> Self {
        self.residual_placement = residual_placement;
        self
    }

    /// Creates a fresh sibling [`PartialEvaluationContext`] that folds pure known work into this context's parent
    /// and retains every effectful operation in its residual program, even when all operands are known. Each residual
    /// invocation therefore executes its own effects, including fresh reference allocations. The shared parent identity
    /// permits explicit known-value transfers between these contexts without relying on constant-value resolution.
    /// The sibling preserves this context's reference placement, including staging folded reference accesses under
    /// an eager parent.
    #[inline]
    pub fn deferred_sibling(&self) -> Self {
        Self::from_shared_parent(self.parent.clone())
            .with_reference_placement(self.reference_placement)
            .with_allow_effect_folding(false)
            .with_residual_placement(self.residual_placement.clone())
    }

    /// Returns the known-side parent [`Context`] of this [`PartialEvaluationContext`] which is used
    /// to fold known subcomputations.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ReferencePlacement`] of known reference operations under an eager known-side parent for this
    /// [`PartialEvaluationContext`]. That placement specifies whether known reference operations execute or remain
    /// residual when the parent is eager. Specialization retains these operations so they observe state when the
    /// specialized program is called.
    #[inline]
    pub fn reference_placement(&self) -> ReferencePlacement {
        self.reference_placement
    }

    /// Returns whether effectful operations with known inputs may run in the parent [`Context`] of this
    /// [`PartialEvaluationContext`]. Refer to [`with_allow_effect_folding`](Self::with_allow_effect_folding)
    /// for more information.
    #[inline]
    pub fn allow_effect_folding(&self) -> bool {
        self.allow_effect_folding
    }

    /// Returns whether ordered effects must currently remain residual to preserve execution order, which is the case
    /// once an ordered operation has been staged (or has failed) in this context or in a clone that shares its flag.
    #[inline]
    pub(super) fn defer_ordered_effects(&self) -> bool {
        self.defer_ordered_effects.get()
    }

    /// Returns the residual placement of this [`PartialEvaluationContext`], if a residual policy was set (refer to
    /// [`with_residual_policy`](Self::with_residual_policy)).
    #[inline]
    pub(super) fn residual_placement(&self) -> Option<Rc<dyn ResidualPlacement<C::Constant, C::Operation>>> {
        self.residual_placement.clone()
    }

    /// Returns the first binding error that this [`PartialEvaluationContext`] retained for finalization, if any (refer
    /// to [`fold_or_residualize`](Self::fold_or_residualize) for more information). Rules that perform work outside of
    /// the context (e.g., by interpreting a program in its [`parent`](Self::parent)) report this error before doing so,
    /// as binding through the context does.
    #[inline]
    pub(crate) fn error(&self) -> Option<ProgramError> {
        self.error.borrow().clone()
    }

    /// Imports a known value from another [`PartialEvaluationContext`] sharing this context's parent. The first import
    /// gets a fresh residual materialization slot. Repeated imports reuse that slot, preserving input/constant intent
    /// without copying source atom identifiers. Values already belonging to this context are returned unchanged,
    /// including unknown values. Foreign unknown values and different parent identities are rejected. A deferred
    /// binding failure instead produces a poisoned value in this context and is retained until finalization, preserving
    /// infallible operator syntax. Importing a reference preserves its live handle; it does not snapshot the referenced
    /// contents.
    pub fn import_known(&self, value: &PartialTracer<C>) -> Result<PartialTracer<C>, ProgramError> {
        let error = self
            .error
            .borrow()
            .clone()
            .or_else(|| value.context.error.borrow().clone())
            .or_else(|| value.value().err());
        if let Some(error) = error {
            self.error.borrow_mut().get_or_insert_with(|| error.clone());
            return Ok(PartialTracer::poisoned(self.clone(), error, value.r#type().into_owned()));
        }
        let partial = value.value()?;
        if Rc::ptr_eq(&self.builder, &value.context.builder) {
            return Ok(value.clone());
        }
        if !Rc::ptr_eq(&self.parent, &value.context.parent) {
            return Err(ProgramError::MalformedProgram(
                "cannot import a value from a partial-evaluation context with a different parent context".to_string(),
            ));
        }
        let known = partial.as_known().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "cannot import an unknown value from another partial-evaluation context".to_string(),
            )
        })?;
        let source_key = Rc::as_ptr(&partial.materialization) as usize;
        if let Some((_, imported)) = self.imported_known_values.borrow().get(&source_key) {
            return Ok(PartialTracer::new(self.clone(), imported.clone()));
        }
        let imported = match partial.materialization() {
            PartialValueMaterialization::Constant { .. } => PartialEvaluationValue::known_constant(known.clone()),
            PartialValueMaterialization::Input { .. } => PartialEvaluationValue::known_input(known.clone()),
            PartialValueMaterialization::Undecided => PartialEvaluationValue::known(known.clone()),
            PartialValueMaterialization::Variable { .. } => unreachable!("known values never name residual variables"),
        };
        self.imported_known_values
            .borrow_mut()
            .insert(source_key, (partial.materialization.clone(), imported.clone()));
        Ok(PartialTracer::new(self.clone(), imported))
    }

    /// Creates a fresh unknown value backed by a new residual-program input of the provided type, recorded as an
    /// [`Unknown`](PartialEvaluationInput::Unknown) feeder carrying `index`. This is how drivers seed the unknown
    /// inputs of an evaluation. The program-replay driver behind [`Program::partially_evaluate_in_context`] seeds one
    /// per unknown program input (with the original program input index as the ordinal), and closure drivers seed one
    /// per traced unknown (e.g., one tangent input per primal in linearization), in order.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`Type`] of the unknown value.
    ///   - `index`: Index recorded in the resulting [`Unknown`](PartialEvaluationInput::Unknown) feeder, which
    ///     [`PartialEvaluation::interpret`] uses to align runtime values with unknown feeders.
    #[inline]
    pub fn unknown_input(&self, r#type: C::Type, index: usize) -> PartialEvaluationValue<C::Value> {
        let atom = self.builder.borrow_mut().add_input(r#type.clone());
        self.inputs.borrow_mut().push(PartialEvaluationInput::Unknown(index));
        PartialEvaluationValue::variable(r#type, atom)
    }

    /// Returns whether the provided effects may execute in the known-side context at this point. Reference placement
    /// and input knowledge impose additional constraints in [`Self::fold_or_residualize`].
    #[inline]
    pub fn can_fold_effects(&self, effects: EffectClasses) -> bool {
        (self.allow_effect_folding || effects.is_empty())
            && (!effects.is_ordered() || !self.defer_ordered_effects.get())
    }

    /// Applies the default partial-evaluation policy to the provided `operation`. When all inputs are known, the
    /// operation is [`bind`](Context::bind)ed in the known-side [`Context`] (i.e., interpreting it under an eager
    /// context and staging it into the outer program under a [`StagingContext`]), and its outputs become known trace
    /// values. When any input is residual, all inputs are materialized into the residual program and the operation is
    /// emitted unchanged.
    ///
    /// Before that execution policy, regionless operations without references can use [`Operation::fold`] to return
    /// existing inputs, even when unknown, or known constants determined by their inferred output types. Those
    /// substitutions prove that the operation has no observable behavior to execute; they preserve input
    /// materialization and do not alter previously recorded effect ordering.
    ///
    /// An operation that carries deferred work, directly or in one of its executable computation regions, is always
    /// residualized, even when all of its inputs are known, because folding it would discharge its unresolved
    /// transformation obligation (refer to the [Deferred Work](crate::Effects#deferred-work) section of the
    /// [`Effects`](crate::Effects) documentation). Dormant rule regions do not contribute deferred work.
    /// Rules that call this function therefore need not check deferred work themselves.
    ///
    /// # Effect Placement Contract
    ///
    /// Known inputs permit folding only when doing so preserves effect order. Once an ordered operation is deferred,
    /// later ordered operations remain residual so they cannot run ahead of it, even when their inputs are known.
    ///
    /// An eager parent attempts to execute known operations. A pure regionless operation without reference inputs
    /// or outputs remains residual if execution returns [`ProgramError::UnsupportedOperation`], allowing execution
    /// environments that support it to run it later. Other failures propagate. [`ReferencePlacement::Stage`]
    /// residualizes every operation touching references, including allocation and aliasing, so eager specialization
    /// cannot observe or mutate live reference state. [`ReferencePlacement::Execute`] permits known reference
    /// operations to execute. A staging parent appends folded operations to its outer program, which executes before
    /// the residual program; reference placement adds no restriction there. A deferred sibling still retains all
    /// effects regardless of whether its parent is eager or staged.
    ///
    /// Fixed-point probes must never fold effectful programs through the live parent context as each probe would
    /// execute or stage the effects again. Control flow rules instead retain such programs whole, without changing
    /// the active context's ordering state during speculative analysis.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] to fold into the known-side context when all inputs are known, or to emit into
    ///     the residual [`Program`] otherwise.
    ///   - `regions`: Owned [`Program`]s whose entry [`Region`](crate::Region)s are attached to `operation`, in
    ///     the order defined by [`Operation::region_slots`]. Folding binds these regions with the operation, while
    ///     residualization imports them into the residual [`Program`].
    ///   - `inputs`: Partially evaluated inputs/operands supplied to `operation`, in [`Operation`]-defined order.
    ///     Their known-ness determines whether the operation is folded or residualized.
    pub fn fold_or_residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if let Some(error) = self.error() {
            return Err(error);
        }

        let operation = operation.into();

        // Specifies whether this operation may stay in the residual program when an eager parent reports that it cannot
        // execute it (i.e., returns `ProgramError::UnsupportedOperation`), rather than failing partial evaluation. This
        // is how work that has no eager value on this backend, such as a reduction over a manual mesh axis, survives
        // for an execution environment that supports it. The flag starts as `false` and becomes `true` only after the
        // validation below proves that the operation is regionless and that no input, output, or effect declaration
        // involves references as deferring such an operation cannot skip observable state changes. The eager binding
        // further down also requires the operation to have no effects before it acts on this flag.
        let mut can_defer_unsupported = false;

        // Reference and region applications retain their existing binding and failure-ordering paths. Local
        // regionless folds resolve before substitution: an input replacement preserves that input's shared
        // materialization slot, and a singleton replacement becomes a known constant of the inferred output type.
        if regions.is_empty() && !operation.effects().has_reference_declarations() {
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            if !input_types.iter().any(Type::is_reference) {
                let replacements = (|| -> Result<_, ProgramError> {
                    operation.validate_region_count(0)?;
                    let output_types = operation.infer_output_types(&input_types, &[])?;
                    operation.effects().validate_application(operation.name(), &input_types, &output_types)?;
                    can_defer_unsupported = !output_types.iter().any(Type::is_reference);
                    Ok(operation.resolve_fold::<C::Constant>(&input_types, &[], &output_types)?)
                })()
                .inspect_err(|_| {
                    // A failed ordered application must not let later known effects run ahead of the failed work.
                    if operation.effects().classes().is_ordered() {
                        self.defer_ordered_effects.set(true);
                    }
                })?;
                if let Some(replacements) = replacements {
                    return replacements
                        .into_iter()
                        .map(|replacement| match replacement {
                            OperationFoldReplacement::Input(index) => Ok(inputs[index].clone()),
                            OperationFoldReplacement::Constant(constant) => {
                                Ok(PartialEvaluationValue::known_constant(self.parent.lift(constant)?))
                            }
                        })
                        .collect();
                }
            }
        }

        // Combine the operation's effects with those of its executable computation regions. Dormant derivative rules
        // and other non-computation regions do not contribute effects or deferred work when this operation runs.
        let summary = regions
            .iter()
            .enumerate()
            .filter(|(index, _)| operation.region_role(*index) == Some(RegionRole::Computation))
            .fold(operation.effects().summary(), |effects, (_, region)| effects.union(region.effects()));
        let effects = summary.classes();

        // Check reference placement only if the cheaper conditions have not already required residual execution.
        // Under eager specialization, reference accesses and root forwarding must remain residual so they do not
        // observe live state before the specialized program runs. Dormant rule regions do not execute here.
        let must_defer_references = || {
            self.reference_placement == ReferencePlacement::Stage
                && self.parent.is_eager()
                && (inputs.iter().any(|input| input.r#type().is_reference())
                    || operation.effects().has_reference_declarations()
                    || regions.iter().enumerate().any(|(index, region)| {
                        operation.region_role(index) == Some(RegionRole::Computation)
                    // Checked reference-effect endpoints are reference-typed, so inspecting atoms covers
                    // allocations, accesses, and aliases without rescanning their declarations.
                    && region.entry_region_ref().computation_regions().any(|region| {
                    region.atoms().iter().any(|atom| atom.r#type().is_reference())
                })
                    }))
        };

        // Deferred work always remains residual because folding it would discharge its transformation obligation.
        if !inputs.iter().all(PartialEvaluationValue::is_known)
            || summary.has_deferred_work()
            || !self.can_fold_effects(effects)
            || must_defer_references()
        {
            return self.residualize_with_effects(operation, regions, inputs, effects);
        }

        let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
        let outputs = if self.parent.is_eager() && can_defer_unsupported && effects.is_empty() {
            // Unsupported pure kernels can remain residual without repeating observable work. The validation above
            // excludes references and regions; execution errors other than unsupported kernels remain failures.
            match self.parent.bind(operation.clone(), Vec::new(), &known) {
                Err(ProgramError::UnsupportedOperation { .. }) => {
                    return self.residualize_with_effects(operation, regions, inputs, effects);
                }
                outputs => outputs?,
            }
        } else {
            self.parent.bind(operation, regions, &known)?
        };

        Ok(outputs.into_iter().map(|value| self.folded_value(value)).collect())
    }

    // TODO(eaplatanios): Should this be renamed to `lift`, moved to above `fold_or_residualize`, and made public?
    /// Returns the provided value, which work folded into the known-side [`Context`] produced, as a known
    /// [`PartialEvaluationValue`]. A folded value that owns a type identity must remain a producer when it crosses
    /// into residual work, which embedding its cheap constant payload does structurally. Symbolic known values remain
    /// residual inputs because their known-side producer stays live.
    pub(crate) fn folded_value(&self, value: C::Value) -> PartialEvaluationValue<C::Value> {
        let defines_identity =
            value.r#type().identities().any(|(position, _)| position == TypeIdentityPosition::Definition);
        if defines_identity && self.parent.resolve(&value).is_constant() {
            PartialEvaluationValue::known_constant(value)
        } else {
            PartialEvaluationValue::known(value)
        }
    }

    /// _Residualizes_ the provided [`Operation`] into the residual [`Program`], materializing each known input into
    /// a residual program [`Atom`](crate::Atom) according to its [`PartialValueMaterialization`], and returns the
    /// operation's outputs as [`PartialEvaluationValue`]s, in output order. Materializing a known value deduplicates
    /// it two ways so a value consumed by several residualized [`Instruction`](crate::Instruction)s yields one
    /// residual input (or inline constant): through the value's shared [`PartialValueMaterialization`] slot, which
    /// records the residual atom assigned on first materialization and is visible to every clone of the value, and,
    /// for inputs, by its *staged* identity across the whole evaluation when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in the known-side context. A
    /// [`Constant`](PartialValueMaterialization::Constant) materialization is only ever attached to values that
    /// originated as literals (i.e., replayed-program constants lifted into the known-side context, or rule-produced
    /// [`known_constant`](PartialEvaluationValue::known_constant) values), and so recovering its payload through
    /// [`Context::resolve`] is expected to succeed. This is what keeps the residual program in the staged-constant
    /// space, since under a staging known-side context a known value is a [`Tracer`](crate::Tracer) that can never
    /// itself be a residual-program constant. A known reference-typed input value materializes as a residual input
    /// (i.e., a residual reference) holding the live handle itself, while a reference-typed program constant (e.g., a
    /// captured reference) follows its [`Constant`](PartialValueMaterialization::Constant) materialization and embeds
    /// inline like every other constant, so it is never reported by [`PartialEvaluation::known_reference_inputs`].
    ///
    /// Note that emitting an ordered operation records that later ordered effects must remain residual too. This also
    /// applies to rules that emit residual work directly (e.g., the residual half of a partitioned boundary), so those
    /// rules preserve the same execution order as the default partial evaluation rule.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] to emit into the residual [`Program`].
    ///   - `regions`: Owned [`Program`]s whose entry [`Region`](crate::Region)s are attached to `operation`, in the
    ///     order defined by [`Operation::region_slots`]. These programs are imported into the residual [`Program`]
    ///     before the operation is emitted.
    ///   - `inputs`: Partially evaluated inputs/operands supplied to `operation`, in [`Operation`]-defined order. Known
    ///     inputs are materialized as residual inputs or constants, while unknown inputs reuse their existing residual
    ///     atoms.
    pub fn residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if let Some(error) = self.error() {
            return Err(error);
        }

        let operation = operation.into();

        // Combine the operation's effect classes with those of its executable computation regions. Dormant
        // derivative rules and other non-computation regions do not contribute effects when this operation runs.
        let effects = regions
            .iter()
            .enumerate()
            .filter(|(index, _)| operation.region_role(*index) == Some(RegionRole::Computation))
            .fold(operation.effects().summary(), |effects, (_, region)| effects.union(region.effects()))
            .classes();

        self.residualize_with_effects(operation, regions, inputs, effects)
    }

    /// Residualizes `operation` using its already computed [`EffectClasses`]. This is the emission behind
    /// [`Self::residualize`] and the residual branch of [`Self::fold_or_residualize`].
    fn residualize_with_effects(
        &self,
        operation: C::Operation,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
        effects: EffectClasses,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Record the ordering restriction before emission, which may fail partway through. Even if binding retains that
        // failure as poisoned outputs, later ordered operations must not execute ahead of the failed work. Retaining
        // the restriction without an emitted instruction can only defer additional ordered operations.
        if effects.is_ordered() {
            self.defer_ordered_effects.set(true);
        }

        // Materialize each known input into a residual-program atom. The deduplication fast-paths return early,
        // and a genuine error rides `?` out through the `collect` into `residualize`.
        let input_atoms = inputs
            .iter()
            .map(|input| -> Result<AtomId, ProgramError> {
                // A residual variable is already a residual atom, and an already-materialized known value's shared
                // slot carries the residual atom assigned on first materialization. Every other known value differs
                // only in whether it materializes as an inline constant.
                let constant = match input.materialization() {
                    PartialValueMaterialization::Undecided => false,
                    PartialValueMaterialization::Input { residual_atom: None } => false,
                    PartialValueMaterialization::Input { residual_atom: Some(atom) } => return Ok(atom),
                    PartialValueMaterialization::Constant { residual_atom: None } => true,
                    PartialValueMaterialization::Constant { residual_atom: Some(atom) } => return Ok(atom),
                    PartialValueMaterialization::Variable { residual_atom } => return Ok(residual_atom),
                };

                let known = input.as_known().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "residual materialization marked an unknown value as a known residual".to_string(),
                    )
                })?;

                let atom = if constant {
                    let constant = self.parent.resolve(known).into_constant().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "residual materialization required a constant payload for a known value that is \
                             not resolvable to a constant in the active known-side context"
                                .to_string(),
                        )
                    })?;
                    let reference_identity =
                        known.r#type().is_reference().then(|| self.parent.reference_identity(known)).transpose()?;
                    let atom = self.builder.borrow_mut().add_constant(constant);
                    if let Some(key) = reference_identity {
                        self.constant_reference_identities.borrow_mut().insert(atom, key);
                    }
                    atom
                } else {
                    // Only input feeders deduplicate across distinct values naming the same known-side staged atom.
                    let staged_atom = match self.parent.resolve(known) {
                        ValueResolution::Staged(atom) => Some(atom),
                        _ => None,
                    };
                    let existing = staged_atom.and_then(|staged| self.staged_feeders.borrow().get(&staged).copied());
                    if let Some(atom) = existing {
                        atom
                    } else {
                        let atom = self.builder.borrow_mut().add_input(known.r#type().into_owned());
                        self.inputs.borrow_mut().push(PartialEvaluationInput::Known(known.clone()));
                        if let Some(staged_atom) = staged_atom {
                            self.staged_feeders.borrow_mut().insert(staged_atom, atom);
                        }
                        atom
                    }
                };

                // Record the assignment in the value's shared slot so that every clone of this value reuses the same
                // residual atom instead of materializing the value again.
                input.materialization.set(match constant {
                    true => PartialValueMaterialization::Constant { residual_atom: Some(atom) },
                    false => PartialValueMaterialization::Input { residual_atom: Some(atom) },
                });
                Ok(atom)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        // Residualized regions splice into the residual builder's arena directly (i.e., owned move), in region order.
        let region_ids = {
            let mut builder = self.builder.borrow_mut();
            regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>()
        };
        let output_atoms = self
            .builder
            .borrow_mut()
            .add_instruction(operation, region_ids, input_atoms, Some(self.provenance.current()))?
            .to_vec();

        let builder = self.builder.borrow();
        Ok(output_atoms
            .into_iter()
            .map(|atom| {
                let r#type = builder.atoms()[atom.index()].r#type().into_owned();
                PartialEvaluationValue::variable(r#type, atom)
            })
            .collect())
    }

    /// Replays the provided [`Program`] through this context using the provided `inputs` bound to its input
    /// [`Atom`](crate::Atom)s in input order, and returns the replay value of each program output, in output order.
    /// The replay is [`Program::interpret_with`] instantiated at this context's protocol (i.e., the same one
    /// [`bind`](Context::bind) applies operation by operation). Each live program constant is [`lift`](Context::lift)ed
    /// into the known-side context as an inline-constant known (rebuilt in the residual program through
    /// [`Context::resolve`] if residual work consumes it), and each [`Instruction`](crate::Instruction) dispatches to
    /// its [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, folding all-known work through the
    /// known-side [`Context`] and emitting residual work into this context's residual [`ProgramBuilder`].
    /// [`Operation`]-specific rules can call this function to recursively replay nested programs over selected inputs,
    /// so that an operation can rewrite itself into transformed work. For example, a known-predicate `condition` can
    /// inline its selected branch.
    ///
    /// # Parameters
    ///
    ///   - `program`: [`Program`] whose entry [`Region`](crate::Region) is replayed through this
    ///     [`PartialEvaluationContext`].
    ///   - `inputs`: Partially evaluated values bound to `program`'s input [`Atom`](crate::Atom)s in input order.
    #[inline]
    pub fn inline_program(
        &self,
        program: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        self.inline_region(program.entry_region_ref(), inputs, &HashSet::new(), None, None)
    }

    /// Replays a borrowed [`Region`](crate::Region) without materializing it as a standalone source [`Program`].
    /// Input binding and output ordering follow [`inline_program`](Self::inline_program). Each call tracks its own
    /// source instruction indices, including recursive calls into shared regions.
    ///
    /// # Parameters
    ///
    ///   - `region`: Borrowed region to replay through this context.
    ///   - `inputs`: Partially evaluated values bound to the region's input atoms in input order.
    ///   - `deferred_instructions`: Region-qualified instruction IDs that must emit residual work regardless of input
    ///     knownness. Replay checks IDs in the source region, before rebuilding instructions in the residual program.
    ///   - `residual_instructions`: Optional vector to which replay appends the IDs of source instructions that emit
    ///     residual work or produce unknown outputs, in source order. This includes instructions with no outputs.
    ///     Recording these observations does not affect placement.
    ///   - `reference_analysis`: Canonical source reference analysis for replay with repeated residual calls. When
    ///     absent, replay preserves ordering across all ordered effects rather than separating independent references.
    pub(super) fn inline_region(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
        deferred_instructions: &HashSet<InstructionId>,
        residual_instructions: Option<&RefCell<Vec<InstructionId>>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let deferred_effect_ordering = RefCell::new(EffectOrdering::default());
        let instruction_index = Cell::new(0);
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(PartialEvaluationValue::known_constant(self.parent.lift(constant.clone())?)),
            |instruction, inputs| {
                // Evaluate inside the source instruction's recorded origin so that residualized instructions record
                // where they came from.
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let index = instruction_index.get();
                instruction_index.set(index + 1);
                let driver = RecursivePartialEvaluationDriver {
                    driver: &regions,
                    repeated_residual: reference_analysis.is_some(),
                };

                // Count emitted instructions only when recording observations. Output knownness alone cannot detect
                // residual effects with no outputs, or an operation that emits effects and returns known outputs.
                let residual_instruction_count =
                    residual_instructions.map(|_| self.builder.borrow().instructions().len());

                let outputs = self.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    driver.partially_evaluate_instruction(
                        self,
                        region,
                        index,
                        inputs,
                        deferred_instructions,
                        &deferred_effect_ordering,
                        reference_analysis,
                    )
                })?;

                // Record the source instruction if it produces unknown outputs or emits residual work. Checking
                // builder growth also catches effects with no outputs and rules that emit effects but return known
                // values. The initial count exists whenever observations are requested, so the `unwrap` is safe.
                if let Some(residual_instructions) = residual_instructions
                    && (outputs.iter().any(PartialEvaluationValue::is_unknown)
                        || self.builder.borrow().instructions().len() > residual_instruction_count.unwrap())
                {
                    residual_instructions.borrow_mut().push(InstructionId::new(region.id(), index));
                }

                Ok(outputs)
            },
        )
    }

    /// Inlines a [`PartitionedProgram`] into this walk as two boundary operations, consuming the partitioned program
    /// and returning the reassembled original boundary outputs, in original output order. This is the shared emission
    /// protocol of online boundary partial-evaluation rules (i.e., the boundary-wise counterpart of the
    /// instruction-wise [`inline_program`](Self::inline_program)). The partitioned program's known [`Program`] is
    /// wrapped through `build_known_operation` and [folded-or-residualized](Self::fold_or_residualize) over the
    /// original known boundary inputs. The residual program is wrapped through `build_residual_operation` and
    /// [residualized](Self::residualize) over the surviving unknown boundary inputs plus the known-side operation's
    /// residual outputs. Each original output is picked from the known-side or residual-side operation's outputs per
    /// the partitioned program's [`outputs`](PartitionedProgram::outputs). Consuming the partitioned program in a
    /// single step keeps it whole until it is gone, so no partially moved partition state can ever be observed.
    ///
    /// # Parameters
    ///
    ///   - `partition`: [`PartitionedProgram`] to inline, produced by [`Program::partition`].
    ///   - `inputs`: Input [`PartialEvaluationValue`]s in the order of the original program's input, pre-partitioning.
    ///   - `build_known_operation`: Wraps the provided known [`Program`] in the known-side boundary [`Operation`]
    ///     together with the owned [`Region`](crate::Region) programs (in region order) that the emitted instruction
    ///     attaches. Operations that carry the program in their payload return no regions.
    ///   - `build_residual_operation`: Wraps the provided residual [`Program`] in the residual boundary [`Operation`],
    ///     with the same [`Region`](crate::Region) contract as `build_known_operation`.
    pub fn inline_partitioned_program<
        V: Value<Type = C::Type>,
        O: Operation<Type = C::Type>,
        P: Into<C::Operation>,
        BuildKnownProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> (P, Vec<FlatProgram<C>>),
        BuildResidualProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> (P, Vec<FlatProgram<C>>),
    >(
        &self,
        program: PartitionedProgram<V, O>,
        inputs: &[PartialEvaluationValue<C::Value>],
        build_known_operation: BuildKnownProgramOperation,
        build_residual_operation: BuildResidualProgramOperation,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Bind the known-side operation into the known-side context over the original known inputs.
        let (known_program, residual_program, known_input_indices, residual_inputs, outputs) = program.into_parts();
        let known_inputs = known_input_indices
            .iter()
            .map(|&index| {
                inputs
                    .get(index)
                    .cloned()
                    .ok_or(ProgramError::InvalidInputCount { expected: index + 1, actual: inputs.len() })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (known_program_operation, known_regions) = build_known_operation(known_program);
        let known_outputs =
            self.fold_or_residualize(known_program_operation, known_regions, known_inputs.as_slice())?;

        // Emit the residual operation over the surviving unknown boundary inputs plus the residual edges, which trail
        // the fully known outputs among the known-side operation's outputs. The emission is unconditional: a residual
        // program without outputs can still carry effectful residual instructions whose effects must be preserved, and
        // an entirely empty residual program only yields a dead pure operation that the walk's final simplification
        // removes.
        let known_output_count = outputs.iter().filter(|output| output.is_known()).count();
        let residual_inputs = residual_inputs
            .iter()
            .map(|source| match source {
                PartialEvaluationInput::Unknown(index) => inputs
                    .get(*index)
                    .cloned()
                    .ok_or(ProgramError::InvalidInputCount { expected: *index + 1, actual: inputs.len() }),
                PartialEvaluationInput::Known(index) => {
                    known_outputs.get(known_output_count + index).cloned().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "known program partition produced no output for residual known input index {index}",
                        ))
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (residual_program_operation, residual_regions) = build_residual_operation(residual_program);
        let residual_outputs =
            self.residualize(residual_program_operation, residual_regions, residual_inputs.as_slice())?;

        // Reassemble the original outputs from the two operations' outputs.
        outputs
            .iter()
            .map(|source| match source {
                PartialEvaluationOutput::Known(index) => known_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "known program partition produced no output for known output {index}",
                    ))
                }),
                PartialEvaluationOutput::Unknown(index) => residual_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "residual program partition produced no output for residual output {index}",
                    ))
                }),
            })
            .collect()
    }

    /// Recovers the staged-constant payload of the provided known value `value` through [`Context::resolve`], reporting
    /// a [`ProgramError`] when the known-side [`Context`] cannot resolve the provided value to a program constant.
    /// Higher-order rules use this when they must embed a known value *inside* a nested residual program (e.g., a
    /// folded loop-invariant carry spliced into a rebuilt `scan` body), where only a program constant can represent
    /// it (nested programs cannot reference atoms of the enclosing residual program or of the outer known-side
    /// program). Under an eager known-side context this always succeeds. Under a [`StagingContext`] it succeeds only
    /// for literal-backed values, and the caller must treat the error as "this rewrite is not available" and fall back
    /// to a conservative alternative.
    #[inline]
    pub fn known_constant(&self, value: &C::Value) -> Result<C::Constant, ProgramError> {
        self.parent.resolve(value).into_constant().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "a known value crossing into a nested residual program does not resolve to a constant in the active \
                 known-side context"
                    .to_string(),
            )
        })
    }

    /// Returns `true` if every [`Known`](PartialEvaluationInput::Known) residual input and
    /// [`Known`](PartialEvaluationOutput::Known) output of the provided [`PartialEvaluation`] resolves to a program
    /// constant in the known-side [`Context`] of this [`PartialEvaluationContext`] (i.e., if a nested program rebuild
    /// that embeds those knowns as inline program constants through [`Self::known_constant`] can succeed). Under a
    /// staging known-side context, a probe's folds can produce known values that are genuine tracers into the live
    /// trace (e.g., a constant-only chain staged by the fold). Rules that rebuild nested programs from a live context
    /// probe must check this and fall back to a conservative rewrite when it returns `false`.
    #[inline]
    pub fn all_knowns_are_constants(&self, evaluation: &PartialEvaluation<C>) -> bool {
        evaluation.inputs.iter().all(|input| match input {
            PartialEvaluationInput::Known(value) => self.parent.resolve(value).is_constant(),
            PartialEvaluationInput::Unknown(_) => true,
        }) && evaluation.outputs.iter().all(|output| match output {
            PartialEvaluationOutput::Known(value) => self.parent.resolve(value).is_constant(),
            PartialEvaluationOutput::Unknown(_) => true,
        })
    }

    /// Returns `true` when any of the provided `inputs` is known but does not [`resolve`](Context::resolve)
    /// to a [`Constant`](ValueResolution::Constant) in the known-side [`Context`] of this
    /// [`PartialEvaluationContext`] (i.e., it is a genuine [`Tracer`](crate::Tracer) into a live outer trace). This
    /// is the signal online boundary rules split on: all-constant knowledge keeps the default fold-or-residualize
    /// behavior.
    #[inline]
    pub fn any_known_is_symbolic(&self, inputs: &[PartialEvaluationValue<C::Value>]) -> bool {
        inputs.iter().any(|input| match input.value() {
            PartialValue::Known(value) => !self.parent.resolve(value).is_constant(),
            PartialValue::Unknown(_) => false,
        })
    }

    /// Consumes this [`PartialEvaluationContext`] and finalizes it into a [`PartialEvaluation`] whose outputs are the
    /// provided evaluation values: known outputs fold to their carried values, unknown outputs become the residual
    /// program's outputs in order, and the accumulated residual program is built and simplified. This is the shared
    /// epilogue of every partial-evaluation driver (the program-replay entry points and the closure-driven trace).
    /// Any deferred binding failure is returned first, even if the failed operation's outputs were discarded.
    /// Finalization recovers sole ownership of the accumulated residual state, and so every clone of this context
    /// (e.g., contexts stamped on [`PartialTracer`]s) must have been dropped by the time this is called; otherwise
    /// this returns [`ProgramError::EscapedProgramBuilder`], mirroring [`TracingContext`]'s trace boundary.
    ///
    /// # Parameters
    ///
    ///   - `outputs`: [`PartialEvaluationValue`] of each original output, in original output order.
    pub fn into_evaluation(
        self,
        outputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        if let Some(error) = self.error() {
            return Err(error);
        }

        // Assemble outputs. Folded values return directly and residual values index the residual program's outputs.
        let mut evaluation_outputs = Vec::with_capacity(outputs.len());
        let mut residual_output_atoms: Vec<AtomId> = Vec::new();
        for output in outputs {
            let materialization = output.materialization();
            match output.value {
                PartialValue::Known(value) => evaluation_outputs.push(PartialEvaluationOutput::Known(value)),
                PartialValue::Unknown(_) => {
                    let PartialValueMaterialization::Variable { residual_atom } = materialization else {
                        return Err(ProgramError::MalformedProgram(
                            "partial evaluation produced an unknown output without a residual atom".to_string(),
                        ));
                    };
                    evaluation_outputs.push(PartialEvaluationOutput::Unknown(residual_output_atoms.len()));
                    residual_output_atoms.push(residual_atom);
                }
            }
        }

        let output_count = residual_output_atoms.len();
        let inputs = Rc::try_unwrap(self.inputs).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let builder = Rc::try_unwrap(self.builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder
            .build::<Vec<C::Constant>, Vec<C::Constant>>(
                residual_output_atoms,
                vec![Placeholder; inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;
        Ok(PartialEvaluation { program, inputs, outputs: evaluation_outputs })
    }
}

impl<C: Context> Clone for PartialEvaluationContext<C> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            builder: self.builder.clone(),
            inputs: self.inputs.clone(),
            staged_feeders: self.staged_feeders.clone(),
            imported_known_values: self.imported_known_values.clone(),
            constant_reference_identities: self.constant_reference_identities.clone(),
            provenance: self.provenance.clone(),
            reference_placement: self.reference_placement,
            allow_effect_folding: self.allow_effect_folding,
            defer_ordered_effects: self.defer_ordered_effects.clone(),
            residual_placement: self.residual_placement.clone(),
            error: self.error.clone(),
        }
    }
}

impl<C: Context> Domain for PartialEvaluationContext<C> {
    type Type = C::Type;
    type Value = PartialTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context> Context for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<PartialTracer<C>, ProgramError> {
        // Lifting a staged constant produces a known value carrying an inline-constant materialization, so that
        // residual work consuming it rebuilds it as a residual-program constant rather than a residual input.
        Ok(PartialTracer::new(self.clone(), PartialEvaluationValue::known_constant(self.parent.lift(constant)?)))
    }

    fn bind<O: Into<C::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[PartialTracer<C>],
    ) -> Result<Vec<PartialTracer<C>>, ProgramError> {
        // Unwrap the input tracers into context-free partial-evaluation values, dispatch the operation's partial
        // evaluation rule against those, and rewrap the produced values with this context, mirroring how
        // `DifferentiationContext::bind` unwraps to `DifferentiationDual`s and rewraps.
        let operation = operation.into();
        operation.validate_region_count(driver.region_count())?;
        let input_values = match self.error.borrow().clone() {
            Some(error) => Err(error),
            None => inputs
                .iter()
                .map(|input| {
                    if !Rc::ptr_eq(&self.builder, &input.context.builder) {
                        return Err(ProgramError::MalformedProgram(
                            "cannot bind a value belonging to another partial-evaluation context; \
                         import known values explicitly"
                                .to_string(),
                        ));
                    }
                    input.value()
                })
                .collect::<Result<Vec<_>, _>>(),
        };
        let error = match input_values {
            Ok(input_values) => {
                let input_values = input_values.into_iter().cloned().collect::<Vec<_>>();
                let driver = RecursivePartialEvaluationDriver { driver: &driver, repeated_residual: false };
                let outputs = operation.partially_evaluate(self, &driver, input_values.as_slice());
                match outputs {
                    Ok(outputs) => {
                        return Ok(outputs.into_iter().map(|value| PartialTracer::new(self.clone(), value)).collect());
                    }
                    Err(error) => error,
                }
            }
            // A poisoned input means an earlier bind already failed and deferred. Propagate its error.
            Err(error) => error,
        };

        // Retain the first error even when this operation has no outputs or its outputs are discarded. Poison lets
        // infallible operator syntax finish the closure without executing subsequent operations. If output inference
        // also fails, the output arity is unknowable and the error surfaces immediately instead.
        self.error.borrow_mut().get_or_insert_with(|| error.clone());
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let region_interfaces = driver.regions().map(RegionRef::interface).collect::<Vec<_>>();
        let Ok(output_types) = operation.infer_output_types(input_types.as_slice(), region_interfaces.as_slice())
        else {
            return Err(error);
        };

        Ok(output_types
            .into_iter()
            .map(|r#type| PartialTracer::poisoned(self.clone(), error.clone(), r#type))
            .collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // A partial-evaluation context is eager exactly when its known-side inner context is: known values are then
        // concrete, so concretizing extractions (e.g., branching on a known predicate) succeed on the known side,
        // while unknown (residual) values never concretize regardless of the inner context.
        self.parent.is_eager()
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // A `PartialEvaluationContext` is a staging boundary for the residual program. It owns provenance state seeded
        // from its parent instead of delegating reads to the (often terminal eager context) known side.
        self.provenance.current()
    }

    #[inline]
    fn resolve(&self, value: &PartialTracer<C>) -> ValueResolution<C::Constant> {
        // A known value resolves exactly as the known-side inner context resolves its payload (a constant under an
        // eager inner context, staged for live tracers of an enclosing trace), while an unknown value is opaque: it
        // names a residual program variable whose value does not exist until the residual program runs.
        match &value.state {
            PartialTracerState::Live(value) => match value.value() {
                PartialValue::Known(known) => self.parent.resolve(known),
                PartialValue::Unknown(_) => ValueResolution::Opaque,
            },
            PartialTracerState::Poison { .. } => ValueResolution::Opaque,
        }
    }

    fn reference_identity(&self, value: &PartialTracer<C>) -> Result<Option<ReferenceIdentity>, ProgramError> {
        // The default implementation of this function recovers only runtime identities from a value or its resolved
        // constant. Partial evaluation also needs the parent's symbolic identities for known references and the
        // residual builder's allocation identities for unknown references, which resolve as opaque values. This
        // override follows those identities through aliases and checks that the tracer belongs to the correct
        // underlying residual builder.
        if !Rc::ptr_eq(&self.builder, &value.context().builder) {
            return Ok(None);
        }

        // Resolve through the parent for known values and through the residual builder for unknown values. Known input
        // feeders and materialized constants retain their parent identities. Inherited capture constants without
        // recorded parent identities remain unresolved; querying must not lift constants or invent identities.
        let value = value.value()?;
        if !value.r#type().is_reference() {
            return Ok(None);
        }
        match value.value() {
            PartialValue::Known(known) => self.parent.reference_identity(known),
            PartialValue::Unknown(_) => {
                let PartialValueMaterialization::Variable { residual_atom } = value.materialization() else {
                    unreachable!("unknown values always carry a residual variable")
                };
                let builder = self.builder.borrow();
                let (root, constant) = builder.resolve_reference(residual_atom)?;
                if let Some(key) = self.constant_reference_identities.borrow().get(&root) {
                    return Ok(*key);
                }
                if let Some(constant) = constant {
                    return Ok(if constant.capture_index().is_some() {
                        None
                    } else {
                        constant.reference_id().map(ReferenceIdentity::Runtime)
                    });
                }
                if let Some(index) = builder.input_ids().iter().position(|input| *input == root)
                    && let PartialEvaluationInput::Known(known) = &self.inputs.borrow()[index]
                {
                    return self.parent.reference_identity(known);
                }
                builder.reference_identity(residual_atom, Rc::as_ptr(&self.builder) as usize)
            }
        }
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.provenance.invoke_with_origin(origin, function)
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.provenance.invoke_with_scope(scope, function)
    }
}

/// State carried by a [`PartialTracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`PartialEvaluationValue`], or a *poison* recording an error that a failed [`bind`](Context::bind) deferred
/// mirroring [`Tracer`](crate::Tracer)'s poison state. Because the closures driven through a
/// [`PartialEvaluationContext`] use infallible operator sugar with no deferral point of their own,
/// [`bind`](Context::bind) turns its errors into poisoned outputs (and propagates poison from inputs to outputs), so
/// the original error surfaces as a plain `Err` at the evaluation boundary instead of panicking mid-closure. The
/// context also retains the first error independently of these outputs and skips subsequent operations. Unlike
/// [`Tracer`](crate::Tracer)'s poison, the deferred [`ProgramError`] itself is carried, so boundaries report the
/// original failure rather than a generic poison error.
#[derive(Clone)]
pub enum PartialTracerState<C: Context> {
    /// The corresponding [`PartialTracer`] is _live_ and has a corresponding [`PartialEvaluationValue`].
    Live(PartialEvaluationValue<C::Value>),

    /// The corresponding [`PartialTracer`] has been _poisoned_, meaning that it corresponds to an error and
    /// will propagate that error wherever it is used (i.e., it will _poison_ those corresponding downstream
    /// [`PartialTracer`]s too).
    Poison {
        /// [`ProgramError`] that the failed bind deferred.
        error: ProgramError,

        /// [`Type`] of the output the failed bind would have produced.
        r#type: C::Type,
    },
}

/// Value flowing through [`PartialEvaluationContext`]s. This is a [`PartialEvaluationValue`] stamped with the context
/// it flows through, so that closures and transform interpreters can drive partial evaluation directly (it is the
/// closure-facing counterpart of the program-replay driver behind [`Program::partially_evaluate_in_context`]). A known
/// [`PartialTracer`] carries a concrete known-side value (i.e., a concrete value under an eager known-side inner
/// context, and a [`Tracer`](crate::Tracer) into the enclosing trace under a staging one), so concretizing extractions
/// such as [`Concretizable::concretize`](crate::Concretizable::concretize) succeed on it exactly when they succeed on
/// the carried value. This is what lets host control flow branch on known values while partial evaluation is in
/// progress. An unknown [`PartialTracer`] names a residual program variable and carries only its type.
#[derive(Clone)]
pub struct PartialTracer<C: Context> {
    /// [`PartialEvaluationContext`] this value flows through, used to dispatch [`Operation`]s that involve it.
    context: PartialEvaluationContext<C>,

    /// [`PartialTracerState`] of this [`PartialTracer`].
    state: PartialTracerState<C>,
}

impl<C: Context> PartialTracer<C> {
    /// Creates a new live [`PartialTracer`] from a context-free [`PartialEvaluationValue`] and the
    /// [`PartialEvaluationContext`] it flows through.
    #[inline]
    pub fn new(context: PartialEvaluationContext<C>, value: PartialEvaluationValue<C::Value>) -> Self {
        Self { context, state: PartialTracerState::Live(value) }
    }

    /// Creates a poisoned [`PartialTracer`] deferring the provided error. Refer to the documentation of
    /// [`PartialTracerState`] for more information.
    #[inline]
    fn poisoned(context: PartialEvaluationContext<C>, error: ProgramError, r#type: C::Type) -> Self {
        Self { context, state: PartialTracerState::Poison { error, r#type } }
    }

    /// Returns the [`PartialEvaluationContext`] this value flows through.
    #[inline]
    pub fn context(&self) -> &PartialEvaluationContext<C> {
        &self.context
    }

    /// Returns the underlying context-free [`PartialEvaluationValue`], or the deferred error if this value
    /// is poisoned.
    #[inline]
    pub fn value(&self) -> Result<&PartialEvaluationValue<C::Value>, ProgramError> {
        match &self.state {
            PartialTracerState::Live(value) => Ok(value),
            PartialTracerState::Poison { error, .. } => Err(error.clone()),
        }
    }

    /// Consumes this value and returns the underlying context-free [`PartialEvaluationValue`], or the deferred error
    /// if this value is poisoned.
    #[inline]
    pub fn into_value(self) -> Result<PartialEvaluationValue<C::Value>, ProgramError> {
        match self.state {
            PartialTracerState::Live(value) => Ok(value),
            PartialTracerState::Poison { error, .. } => Err(error),
        }
    }
}

// `PartialTracer` equality is *value identity* and not payload equality. Two values are equal if and only if they are
// clones of one logical partial-evaluation value (witnessed by sharing one materialization slot). Two values that would
// evaluate to equal payloads but were produced separately are considered unequal, which is the conservative answer
// analyses such as the scan/while loop-invariance fixed points of partial evaluation need (they degrade to passthrough
// detection, mirroring `Tracer`'s staging-identity `PartialEq`).
impl<C: Context> PartialEq for PartialTracer<C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (&self.state, &other.state) {
            (PartialTracerState::Live(left), PartialTracerState::Live(right)) => {
                Rc::ptr_eq(&left.materialization, &right.materialization)
            }
            // Poisoned values never compare equal: equality answers identity questions for analyses such as the
            // loop-invariance probes, and a deferred error has no identity to assert.
            _ => false,
        }
    }
}

impl<C: Context> Debug for PartialTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            PartialTracerState::Live(value) => formatter.debug_struct("PartialTracer").field("value", value).finish(),
            PartialTracerState::Poison { error, r#type } => {
                formatter.debug_struct("PartialTracer").field("error", error).field("type", r#type).finish()
            }
        }
    }
}

impl<C: Context> Display for PartialTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match &self.state {
            PartialTracerState::Live(value) => value,
            PartialTracerState::Poison { r#type, .. } => return write!(formatter, "<poison:{}>", r#type),
        };
        match (value.value(), value.materialization()) {
            (PartialValue::Known(value), _) => write!(formatter, "{value}"),
            (PartialValue::Unknown(_), PartialValueMaterialization::Variable { residual_atom }) => {
                write!(formatter, "{residual_atom}")
            }
            (PartialValue::Unknown(r#type), _) => write!(formatter, "<unknown:{}>", r#type),
        }
    }
}

impl<C: Context> Typed for PartialTracer<C> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match &self.state {
            PartialTracerState::Live(value) => value.r#type(),
            PartialTracerState::Poison { r#type, .. } => Cow::Borrowed(r#type),
        }
    }
}

impl<C: Context> Parameter for PartialTracer<C> {}

impl<C: Context> Value for PartialTracer<C> {
    type DispatchDomain = PartialEvaluationContext<C>;
    type ExecutionDomain = PartialEvaluationContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> PartialEvaluationContext<C> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> PartialEvaluationContext<C> {
        self.context().clone()
    }
}

impl<C: Context, T: Type> ValueProjection<T> for PartialTracer<C>
where
    for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
{
    type Projected = ProjectedValue<T, Self>;
    type ProjectedRef<'v>
        = ProjectedValue<T, &'v Self>
    where
        Self: 'v,
        T: 'v;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        value.into_value()
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        T: 'v,
    {
        Ok(ProjectedValue::new(self, <&T>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(self.r#type().as_ref())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrType, ArrayOperation, ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArrayType, DataType,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::{Context, EagerContext, StagingContext, ValueResolution};
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::{
        AddOperation, LinearCallOperation, MulOperation, NegOperation, PrintOperation, ReferenceAddUpdateOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation, SubOperation,
        Zero,
    };
    use crate::parameters::Placeholder;
    use crate::partial::evaluations::PartialEvaluation;
    use crate::partial::residuals::{
        NoStorage, ResidualCandidate, ResidualDecision, ResidualPolicy, ResidualRejection,
    };
    use crate::partial::tests::{
        TestCapture, TestOperation, TestValue, reference_ordering_program, replay_reference_ordering_program,
    };
    use crate::partial::values::{
        PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialValue,
        PartialValueMaterialization,
    };
    use crate::programs::{
        AtomId, Concretizable, EffectClass, EffectClasses, Effects, InstructionId, Operation, ProgramBuilder,
        ProgramError, Provenance, ProvenanceScope, ReferenceError, ReferenceIdentity, ReferenceType, RegionInterface,
        TypeError, Typed,
    };
    use crate::tests::{
        TestArrayContext, TestArrayIrContext, TestArrayIrOperation, TestArrayOperation, TestArrayTracingContext,
        TestOrderedStateOperation,
    };
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns a residual policy over [`ArrayIrType`] that recomputes every residual.
    fn save_nothing() -> ResidualPolicyReference<ArrayIrType> {
        struct SaveNothing;

        impl ResidualPolicy<ArrayIrType> for SaveNothing {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "save_nothing"
            }

            fn classify(
                &self,
                _candidate: &ResidualCandidate<'_, ArrayIrType>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(ResidualDecision::Recompute)
            }
        }

        ResidualPolicyReference::new(SaveNothing)
    }

    #[test]
    fn test_partial_evaluation_context_new() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(context.reference_placement(), ReferencePlacement::Execute);
        assert!(context.allow_effect_folding());
        assert!(context.builder.borrow().instructions().is_empty());
        assert!(context.inputs.borrow().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_with_reference_placement() {
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let input = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let configured = context.clone().with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(context.reference_placement(), ReferencePlacement::Execute);
        assert_eq!(configured.reference_placement(), ReferencePlacement::Stage);
        let restored = configured.clone().with_reference_placement(ReferencePlacement::Execute);
        assert_eq!(restored.reference_placement(), ReferencePlacement::Execute);
        assert_eq!(configured.reference_placement(), ReferencePlacement::Stage);
        // Reconfiguration shares the accumulated residual state rather than starting a fresh program.
        let outputs = configured.residualize(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        drop((context, configured));
        let evaluation = restored.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.inputs, vec![PartialEvaluationInput::Unknown(0)]);
        assert_eq!(evaluation.program.instructions().len(), 1);
        let outputs = evaluation
            .interpret(&EagerContext::new(), &[TestValue::Array(Array::scalar(3.0_f32).unwrap())])
            .unwrap();
        assert!(
            matches!(&outputs[0], TestValue::Reference(reference) if reference.read() == Ok(Array::scalar(3.0_f32).unwrap())),
        );
    }

    #[test]
    fn test_partial_evaluation_context_with_allow_effect_folding() {
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let input = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let configured = context.clone().with_allow_effect_folding(false);
        assert!(context.allow_effect_folding());
        assert!(!configured.allow_effect_folding());
        let restored = configured.clone().with_allow_effect_folding(true);
        assert!(restored.allow_effect_folding());
        assert!(!configured.allow_effect_folding());
        // Enabling folding again must restore behavior, while preserving the previously created residual input.
        let initial = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        let allocated =
            restored.fold_or_residualize(ReferenceNewOperation::new(), Vec::new(), &[initial.clone()]).unwrap();
        assert!(
            matches!(allocated[0].as_known(), Some(TestValue::Reference(reference)) if reference.read() == Ok(Array::scalar(2.0_f32).unwrap())),
        );
        let deferred = configured.fold_or_residualize(ReferenceNewOperation::new(), Vec::new(), &[initial]).unwrap();
        assert!(deferred[0].is_unknown());
        drop((configured, restored));
        let evaluation = context.into_evaluation(vec![input, deferred[0].clone()]).unwrap();
        assert_eq!(evaluation.inputs.len(), 2);
        let outputs = evaluation
            .interpret(&EagerContext::new(), &[TestValue::Array(Array::scalar(5.0_f32).unwrap())])
            .unwrap();
        assert_eq!(outputs[0], TestValue::Array(Array::scalar(5.0_f32).unwrap()));
        assert!(
            matches!(&outputs[1], TestValue::Reference(reference) if reference.read() == Ok(Array::scalar(2.0_f32).unwrap())),
        );
    }

    #[test]
    fn test_partial_evaluation_context_with_residual_policy() {
        // Contexts carry no residual policy by default. Configured contexts carry theirs into clones and deferred
        // siblings, while existing clones keep their own configuration.
        let policy = save_nothing();
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let configured = context.clone().with_residual_policy(&policy);
        assert!(context.residual_placement().is_none());
        assert!(configured.residual_placement().is_some());
        assert!(configured.clone().residual_placement().is_some());
        assert!(configured.deferred_sibling().residual_placement().is_some());

        // The carried residual placement follows the policy: `f(x, t) = -x * t` with `x` known recomputes
        // the negation in the residual program instead of saving it.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let t = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let negated = builder
            .add_instruction(ArrayOperation::from(NegOperation::new()), Vec::new(), vec![x], None)
            .unwrap()[0];
        let product = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![negated, t], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let planned = configured
            .residual_placement()
            .unwrap()
            .place_residuals(program.partition(&[true, false]).unwrap())
            .unwrap();
        assert_eq!(
            planned.residual_program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = neg %1
                    %3:f32[] = mul %2 %0
                in (%3)"},
        );
    }

    #[test]
    fn test_partial_evaluation_context_without_residual_policy() {
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_residual_policy(&save_nothing());
        let without_policy = context.clone().without_residual_policy();
        assert!(without_policy.residual_placement().is_none());
        assert!(without_policy.deferred_sibling().residual_placement().is_none());
        assert!(context.residual_placement().is_some());
    }

    #[test]
    fn test_partial_evaluation_context_with_defer_ordered_effects() {
        // Clones share their ordered-effect deferral, while a context with its own deferral no longer does.
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let detached = context.clone().with_defer_ordered_effects(true);
        assert!(!context.defer_ordered_effects());
        assert!(detached.defer_ordered_effects());
        context.clone().defer_ordered_effects.set(true);
        assert!(context.defer_ordered_effects());
        let detached = context.clone().with_defer_ordered_effects(false);
        assert!(!detached.defer_ordered_effects());
        assert!(context.defer_ordered_effects());
    }

    #[test]
    fn test_partial_evaluation_context_deferred_sibling() {
        let parent = TracingContext::<TestValue, TestOperation>::new();
        let source = PartialEvaluationContext::new(parent.clone());
        let context = source.deferred_sibling();
        assert!(Rc::ptr_eq(&source.parent, &context.parent));
        assert!(Rc::ptr_eq(&context.parent, &context.clone().parent));
        assert!(!Rc::ptr_eq(&source.builder, &context.builder));
        assert!(!Rc::ptr_eq(&source.parent, &PartialEvaluationContext::new(parent.clone()).parent));
        let zero = context.lift(TestValue::Array(Array::scalar(0.0_f32).unwrap())).unwrap();
        let allocated = context.bind(ReferenceNewOperation::new(), Vec::new(), &[zero.clone()]).unwrap();
        assert!(allocated[0].value().unwrap().is_unknown());
        assert!(parent.builder().borrow().instructions().is_empty());

        // Pure known work can still execute in the parent after a residual effect.
        let folded = context.clone().bind(AddOperation::new(), Vec::new(), &[zero.clone(), zero]).unwrap();
        assert!(folded[0].value().unwrap().is_known());
        assert_eq!(parent.builder().borrow().instructions().len(), 1);
        assert_eq!(context.builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_deferred_sibling_preserves_reference_placement() {
        let source = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        let target = source.deferred_sibling();
        assert_eq!(target.reference_placement(), ReferencePlacement::Stage);
        let reference = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let source_reference = source.lift(TestValue::Reference(reference)).unwrap();
        let target_reference = target.import_known(&source_reference).unwrap();
        let viewed = target
            .bind(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]),
                Vec::new(),
                &[target_reference],
            )
            .unwrap();
        assert!(viewed[0].value().unwrap().is_unknown());
        assert_eq!(target.builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_parent() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(
            context.parent().bind(
                AddOperation::new(),
                Vec::new(),
                &[Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()]
            ),
            Ok(vec![Array::scalar(3.0).unwrap()]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_error() {
        // The first binding failure is retained, and clones share it.
        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        assert_eq!(context.error(), None);
        let reference = context.lift(TestValue::Reference(frozen)).unwrap();
        drop(context.clone().bind(ReferenceReadOperation::new(), Vec::new(), &[reference]).unwrap());
        let error = context.error().unwrap();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_import_known() {
        let parent = TracingContext::<TestValue, TestArrayIrOperation>::new();
        let source = PartialEvaluationContext::new(parent.clone());
        let target = source.deferred_sibling();
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let known =
            PartialTracer::new(source.clone(), PartialEvaluationValue::known(parent.input(scalar_type.clone())));
        source
            .residualize(
                TestArrayOperation::Add(AddOperation::new()),
                Vec::new(),
                &[known.value().unwrap().clone(), known.value().unwrap().clone()],
            )
            .unwrap();
        assert!(matches!(
            known.value().unwrap().materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(_) },
        ),);
        let imported = target.import_known(&known).unwrap();
        assert_eq!(
            imported.value().unwrap().materialization(),
            PartialValueMaterialization::Input { residual_atom: None },
        );
        assert_eq!(target.import_known(&imported).unwrap(), imported);

        let reference = parent.input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let known_reference = PartialTracer::new(source.clone(), PartialEvaluationValue::known(reference));
        let imported_reference = target.import_known(&known_reference).unwrap();
        assert_eq!(target.reference_identity(&imported_reference), source.reference_identity(&known_reference));

        let unknown = PartialTracer::new(source.clone(), source.unknown_input(scalar_type.clone(), 0));
        assert!(matches!(target.import_known(&unknown), Err(ProgramError::MalformedProgram(message))
            if message == "cannot import an unknown value from another partial-evaluation context",),);
        let other_parent = TracingContext::<TestValue, TestArrayIrOperation>::new();
        let foreign = PartialTracer::new(
            PartialEvaluationContext::new(other_parent.clone()),
            PartialEvaluationValue::known(other_parent.input(scalar_type)),
        );
        assert!(matches!(target.import_known(&foreign), Err(ProgramError::MalformedProgram(message))
            if message == "cannot import a value from a partial-evaluation context with a different parent context",),);

        // Binding a foreign value cannot reinterpret its atom identifier in this builder, including in release builds.
        let poisoned = target
            .bind(TestArrayOperation::Add(AddOperation::new()), Vec::new(), &[known.clone(), known])
            .unwrap();
        assert!(matches!(poisoned[0].value(), Err(ProgramError::MalformedProgram(message))
            if message == "cannot bind a value belonging to another partial-evaluation context; import known values explicitly",),);
        assert!(target.builder.borrow().instructions().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_import_known_deduplicates_eager_inputs() {
        let source = PartialEvaluationContext::new(TestArrayContext::new());
        let target = source.deferred_sibling();
        let input =
            PartialTracer::new(source.clone(), PartialEvaluationValue::known_input(Array::scalar(3.0).unwrap()));
        let first = target.import_known(&input).unwrap();
        let second = target.import_known(&input).unwrap();
        assert_eq!(first, second);
        let outputs = target
            .residualize(
                AddOperation::new(),
                Vec::new(),
                &[first.value().unwrap().clone(), second.value().unwrap().clone()],
            )
            .unwrap();
        drop(first);
        drop(second);
        let evaluation = target.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.inputs(), &[PartialEvaluationInput::Known(Array::scalar(3.0).unwrap())]);
        assert_eq!(evaluation.program().inputs().count(), 1);
        assert_eq!(evaluation.interpret(source.parent(), &[]), Ok(vec![Array::scalar(6.0).unwrap()]));
    }

    #[test]
    fn test_partial_evaluation_context_import_known_retains_discarded_errors() {
        let source = PartialEvaluationContext::new(TestArrayContext::new());
        let target = source.deferred_sibling();
        let known = source.lift(Array::scalar(3.0).unwrap()).unwrap();
        assert_eq!(
            source.bind(AddOperation::new(), Vec::new(), &[known.clone()]),
            Err(ProgramError::Type(TypeError::invalid("expected 2 inputs but got 1"))),
        );
        let imported = target.import_known(&known).unwrap();
        assert_eq!(
            imported.value().unwrap_err(),
            ProgramError::Type(TypeError::invalid("expected 2 inputs but got 1")),
        );
        drop(imported);
        assert_eq!(
            target.into_evaluation(Vec::new()).unwrap_err(),
            ProgramError::Type(TypeError::invalid("expected 2 inputs but got 1")),
        );
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        // `fold_or_residualize` folds an all-known operation through the known-side context, and so its outputs
        // are known values with no residual materialization decision yet.
        let inputs = [
            PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
            PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
        ];
        let folded = context.fold_or_residualize(MulOperation::new(), Vec::new(), &inputs).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_known());
        assert!(!folded[0].is_unknown());
        assert_eq!(folded[0].as_known(), Some(&Array::scalar(6.0).unwrap()));
        assert_eq!(folded[0].materialization(), PartialValueMaterialization::Undecided);
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(folded[0].value(), PartialValue::Known(value) if *value == Array::scalar(6.0).unwrap()));

        let unknown = context.unknown_input(ArrayType::scalar(DataType::F64), 0);
        let mixed = context.fold_or_residualize(NegOperation::new(), Vec::new(), &[unknown]).unwrap();
        let evaluation = context.into_evaluation(mixed).unwrap();
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(7.0).unwrap()]),
            Ok(vec![Array::scalar(-7.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_unsupported_eager_execution() {
        /// Operation with a controlled execution failure and effect classification.
        #[derive(Clone, Debug)]
        struct UnavailableOperation {
            /// Error reported by eager execution after successful type inference.
            error: ProgramError,

            /// Observable effects that forbid deferring a failed execution attempt.
            effects: Effects,
        }

        impl Operation for UnavailableOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "unavailable"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                check_count!("region", region_interfaces, 0, TypeError);
                Ok(input_types.to_vec())
            }

            fn effects(&self) -> Cow<'_, Effects> {
                Cow::Borrowed(&self.effects)
            }
        }

        impl InterpretableOperation<EagerContext<Array, Self>> for UnavailableOperation {
            fn interpret<D: InterpretationDriver<EagerContext<Array, Self>>>(
                &self,
                _context: &EagerContext<Array, Self>,
                _driver: &D,
                _inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                Err(self.error.clone())
            }
        }

        let unsupported = ProgramError::UnsupportedOperation { message: "kernel is unavailable".to_string() };
        let operation = UnavailableOperation { error: unsupported.clone(), effects: Effects::empty().clone() };
        let inputs = [PartialEvaluationValue::known(Array::scalar(2.0_f32).unwrap())];
        let context = PartialEvaluationContext::new(EagerContext::<Array, UnavailableOperation>::new());
        let outputs = context.fold_or_residualize(operation.clone(), Vec::new(), &inputs).unwrap();
        assert!(outputs[0].is_unknown());
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::scalar(DataType::F32));
        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert_eq!(evaluation.program.instructions()[0].operation().name(), "unavailable");

        // Staging a known operation does not attempt execution, so its outputs stay known.
        let parent = TracingContext::<Array, UnavailableOperation>::new();
        let input = PartialEvaluationValue::known(parent.input(ArrayType::scalar(DataType::F32)));
        let context = PartialEvaluationContext::new(parent);
        let outputs = context.fold_or_residualize(operation.clone(), Vec::new(), &[input]).unwrap();
        assert!(outputs[0].is_known());
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::scalar(DataType::F32));

        // A failed effectful execution cannot be retried later because it might have already performed effects.
        let effectful = UnavailableOperation {
            effects: Effects::new(EffectClasses::single(EffectClass::OrderedIo), Vec::new()).unwrap(),
            ..operation.clone()
        };
        let context = PartialEvaluationContext::new(EagerContext::<Array, UnavailableOperation>::new());
        assert_eq!(context.fold_or_residualize(effectful, Vec::new(), &inputs).unwrap_err(), unsupported);

        // Unsupported execution does not conceal malformed boundaries or other runtime errors.
        assert_eq!(
            context.fold_or_residualize(operation.clone(), Vec::new(), &[]).unwrap_err(),
            ProgramError::Type(TypeError::invalid("expected 1 input but got 0")),
        );
        let context = PartialEvaluationContext::new(EagerContext::<Array, UnavailableOperation>::new());
        let invalid = ProgramError::Type(TypeError::invalid("invalid kernel input"));
        let operation = UnavailableOperation { error: invalid.clone(), ..operation };
        assert_eq!(context.fold_or_residualize(operation, Vec::new(), &inputs).unwrap_err(), invalid);
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_order_for_unrooted_state() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestOrderedStateOperation>::new());
        let known = [PartialEvaluationValue::known(Array::scalar(1.0).unwrap())];
        let unknown = [context.unknown_input(scalar_type, 0)];

        // Ordered state folds like any other all-known operation before any ordered effect has been deferred.
        let folded = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert_eq!(folded[0].as_known(), Some(&Array::scalar(1.0).unwrap()));

        // A state access with an unknown operand stages. All later ordered operations must then stage too, even
        // with known operands and no shared reference allocation. Pure work keeps folding.
        let staged = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &unknown).unwrap();
        assert!(staged[0].is_unknown());
        let later = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert!(later[0].is_unknown());
        let pure = context.fold_or_residualize(TestOrderedStateOperation::Pure, Vec::new(), &known).unwrap();
        assert_eq!(pure[0].as_known(), Some(&Array::scalar(1.0).unwrap()));
        let evaluation = context.into_evaluation(vec![later[0].clone()]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(1.0).unwrap())],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = state %0
                    %3:f64[] = state %1
                in (%3)
            "}
            .trim_end(),
        );

        // Direct residual emission also records that ordered work was deferred, so later ordered operations
        // remain residual just as they do after the default rule defers an operation.
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestOrderedStateOperation>::new());
        let known = [PartialEvaluationValue::known(Array::scalar(1.0).unwrap())];
        context.residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        let later = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert!(later[0].is_unknown());
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_stages_reference_operations_under_stage_placement() {
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(context.reference_placement(), ReferencePlacement::Stage);
        assert_eq!(
            PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new()).reference_placement(),
            ReferencePlacement::Execute,
        );

        // Under the specialization placement every reference operation stages, all-known operands notwithstanding:
        // the update leaves the live state untouched, the allocation is deferred to the residual program, and only
        // reference-free work (pure or ordered) folds.
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        let initial = PartialEvaluationValue::known(TestValue::Array(Array::scalar(3.0_f32).unwrap()));
        let add_update = TestOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new());
        assert!(
            context
                .fold_or_residualize(add_update, Vec::new(), &[reference.clone(), update])
                .unwrap()
                .is_empty(),
        );
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));
        let allocation = TestOperation::ReferenceNew(ReferenceNewOperation::new());
        let allocated = context.fold_or_residualize(allocation, Vec::new(), &[initial.clone()]).unwrap();
        assert!(allocated[0].is_unknown());
        let product = TestOperation::from(ArrayOperation::from(MulOperation::new()));
        let folded = context.fold_or_residualize(product, Vec::new(), &[initial.clone(), initial.clone()]).unwrap();
        assert_eq!(folded[0].as_known(), Some(&TestValue::Array(Array::scalar(9.0_f32).unwrap())));

        // The live handle crosses into the residual program by identity as a residual reference, so replaying the
        // residual program performs the deferred accesses against the same allocation.
        let evaluation = context.into_evaluation(allocated).unwrap();
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let () = reference_add_update %0 %1
                    %3:ref<f32[]> = reference_new %2
                in (%3)
            "}
            .trim_end(),
        );
        let outputs = evaluation.interpret(&EagerContext::<TestValue, TestOperation>::new(), &[]).unwrap();
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        let TestValue::Reference(allocated) = &outputs[0] else {
            panic!("residual replay returned {:?} instead of the deferred allocation", outputs[0]);
        };
        assert_eq!(allocated.read(), Ok(Array::scalar(3.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_reference_failure_order() {
        use crate::programs::ReferenceError;

        // An unused read can fail, so later I/O must not execute during specialization ahead of that failure.
        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        frozen.freeze().unwrap();
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        let reference = PartialEvaluationValue::known(TestValue::Reference(frozen));
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let output = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        assert!(output[0].is_unknown());
        let value = PartialEvaluationValue::known(TestValue::Array(Array::scalar(42.0_f32).unwrap()));
        let print = TestOperation::from(ArrayOperation::from(PrintOperation::new("after_read")));
        let printed = context.fold_or_residualize(print, Vec::new(), &[value]).unwrap();
        assert!(printed[0].is_unknown());
        let evaluation = context.into_evaluation(printed).unwrap();
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_read", "print"],
        );
        let error = evaluation.interpret(&EagerContext::<TestValue, TestOperation>::new(), &[]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_order_across_reference_roots() {
        use crate::programs::ReferenceError;

        // The default execution policy must also preserve order when the earlier reference is an unknown input.
        // Its failure prevents a later write to another root, including when that write has entirely known operands.
        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        frozen.freeze().unwrap();
        let other = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(TestArrayIrContext::new());
        let reference = context.unknown_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into(), 0);
        let read = TestArrayIrOperation::ReferenceRead(ReferenceReadOperation::new());
        let output = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        let target = PartialEvaluationValue::known(TestValue::Reference(other.clone()));
        let value = PartialEvaluationValue::known(TestValue::Array(Array::scalar(7.0_f32).unwrap()));
        let write = TestArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new());
        assert!(context.fold_or_residualize(write, Vec::new(), &[target, value]).unwrap().is_empty());
        assert_eq!(other.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let evaluation = context.into_evaluation(output).unwrap();
        let error = evaluation.interpret(&TestArrayIrContext::new(), &[TestValue::Reference(frozen)]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(other.read(), Ok(Array::scalar(2.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_excludes_dormant_effects() {
        use crate::programs::RegionSlot;
        use crate::tests::TestRegionOperation;

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let output = builder
            .add_instruction(TestRegionOperation::Effectful(EffectClass::OrderedState), Vec::new(), vec![input], None)
            .unwrap()[0];
        let rule = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let outer = TracingContext::<Array, TestRegionOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone());
        let known = PartialEvaluationValue::known(outer.input(ArrayType::scalar(DataType::F32)));
        let unknown = context.unknown_input(ArrayType::scalar(DataType::F32), 0);
        let operation = TestRegionOperation::WithRegions(const { &[RegionSlot::rule("derivative")] });
        context.fold_or_residualize(operation, vec![rule], &[unknown]).unwrap();
        let retained = context
            .fold_or_residualize(TestRegionOperation::Effectful(EffectClass::OrderedIo), Vec::new(), &[known])
            .unwrap();
        assert!(retained[0].is_known());
        assert_eq!(outer.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_residualizes_deferred_work() {
        use crate::programs::RegionSlot;
        use crate::tests::TestRegionOperation;

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let output = builder.add_instruction(TestRegionOperation::Deferred, Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let outer = TracingContext::<Array, TestRegionOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone());
        let known = PartialEvaluationValue::known(outer.input(ArrayType::scalar(DataType::F32)));

        // Deferred work stays residual even when all inputs are known, whether the operation declares it directly
        // or carries it in an executable computation region, because folding it would discharge its obligation.
        let direct = context.fold_or_residualize(TestRegionOperation::Deferred, Vec::new(), &[known.clone()]).unwrap();
        assert!(direct[0].is_unknown());
        let computation = TestRegionOperation::WithRegions(const { &[RegionSlot::computation("body")] });
        let computation = context.fold_or_residualize(computation, vec![body.clone()], &[known.clone()]).unwrap();
        assert!(computation[0].is_unknown());
        assert_eq!(context.builder.borrow().instructions().len(), 2);
        assert!(outer.builder().borrow().instructions().is_empty());

        // A dormant rule region (i.e., a registered but unselected derivative) creates no obligation,
        // so its application folds into the known-side context.
        let rule = TestRegionOperation::WithRegions(const { &[RegionSlot::rule("derivative")] });
        let rule = context.fold_or_residualize(rule, vec![body], &[known]).unwrap();
        assert!(rule[0].is_known());
        assert_eq!(context.builder.borrow().instructions().len(), 2);
        assert_eq!(outer.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_executes_reference_operations_under_execute_placement() {
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let other = ArrayReference::new(Array::scalar(5.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let other_reference = PartialEvaluationValue::known(TestValue::Reference(other.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let u = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let add_update = TestOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new());
        let write = TestOperation::ReferenceWrite(ReferenceWriteOperation::new());
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());

        // Under the execution placement an all-known reference operation executes at partial-evaluation time like any
        // other known ordered effect. This is the known side of an eager linearization (i.e., the forward pass).
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        assert!(
            context
                .fold_or_residualize(add_update, Vec::new(), &[reference.clone(), update])
                .unwrap()
                .is_empty(),
        );
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        assert!(context.builder.borrow().instructions().is_empty());

        // A write of the unknown value stages and prevents later ordered effects from folding: later known reads stage
        // even on a different allocation, preserving failure and synchronization order.
        assert!(context.fold_or_residualize(write, Vec::new(), &[reference.clone(), u]).unwrap().is_empty());
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        let staged = context.fold_or_residualize(read.clone(), Vec::new(), &[reference]).unwrap();
        assert!(staged[0].is_unknown());
        let other_read = context.fold_or_residualize(read, Vec::new(), &[other_reference]).unwrap();
        assert!(other_read[0].is_unknown());
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_retains_ordering_after_type_error() {
        // A write of an unknown `f64` into an `f32` reference fails at residual emission (type inference rejects the
        // mismatch) after recording that later ordered effects must remain residual, so a later all-known read of that
        // root stages instead of executing against state the failed write attempted to mutate.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let mismatched = context.unknown_input(ArrayType::scalar(DataType::F64).into(), 0);
        let write = TestOperation::ReferenceWrite(ReferenceWriteOperation::new());
        assert!(matches!(
            context.fold_or_residualize(write, Vec::new(), &[reference.clone(), mismatched]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`reference_write` replacement type `f64[]` must exactly \
                               match reference referent type `f32[]`",
        ),);
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let staged = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        assert!(staged[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_residualize() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let inputs = [
            PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
            PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
        ];
        // `residualize` emits the operation into the residual program, materializing each known input as a fresh
        // residual input (i.e., atoms 0 and 1) and returning the instruction output as a residual variable
        // (i.e., atom 2).
        let residual = context.residualize(AddOperation::new(), Vec::new(), &inputs).unwrap();
        assert_eq!(residual.len(), 1);
        assert!(residual[0].is_unknown());
        assert_eq!(residual[0].as_known(), None);
        assert_eq!(residual[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(
            residual[0].materialization(),
            PartialValueMaterialization::Variable { residual_atom: AtomId::new(2) },
        );

        // `fold_or_residualize` residualizes as soon as any input is unknown. `neg` lands in the residual program
        // over the residual variable, producing the next residual atom.
        let mixed = context.fold_or_residualize(NegOperation::new(), Vec::new(), &[residual[0].clone()]).unwrap();
        assert_eq!(mixed[0].materialization(), PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) });

        // Materializing the same known value twice reuses the residual atom assigned on first materialization through
        // the value's shared materialization slot, so a value consumed by several residualized instructions yields a
        // single residual input.
        let shared = PartialEvaluationValue::known_input(Array::scalar(4.0).unwrap());
        let first = context.residualize(NegOperation::new(), Vec::new(), &[shared.clone()]).unwrap();
        let second = context.residualize(AddOperation::new(), Vec::new(), &[shared.clone(), shared.clone()]).unwrap();
        assert_eq!(
            shared.materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(AtomId::new(4)) },
        );
        assert!(first[0].is_unknown() && second[0].is_unknown());
        assert_eq!(context.inputs.borrow().len(), 3);
    }

    #[test]
    fn test_partial_evaluation_context_inline_program() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        // `inline_program` replays a program over seed values. All-known seeds fold every instruction, lifting the
        // program constant into the known-side context, and so the replay returns folded values.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, x], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let outputs = context
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(7.0).unwrap()));

        // Mixed seeds fold the known work and residualize the rest, and so the walk returns residual variables.
        let outputs = context
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    context.unknown_input(ArrayType::scalar(DataType::F64), 0),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::scalar(11.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let observations = RefCell::new(Vec::new());
        let (known, residual) = replay_reference_ordering_program(region, None, None);
        let (observed_known, observed_residual) = replay_reference_ordering_program(region, Some(&observations), None);

        // Recording source instructions must not change ordering or either emitted program. The nested read has no
        // public outputs, so it must be recorded based on the residual instructions it emits.
        assert_eq!(known.to_string(), observed_known.to_string());
        assert_eq!(residual.to_string(), observed_residual.to_string());
        assert_eq!(known.instructions().len(), 0);
        assert_eq!(
            *observations.borrow(),
            vec![InstructionId::new(region.id(), 0), InstructionId::new(region.id(), 1)],
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation_with_reference_analysis() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_configuration(None, true, &[]).unwrap();
        let observations = RefCell::new(Vec::new());
        let (known, residual) = replay_reference_ordering_program(region, None, Some(analysis.as_ref()));
        let (observed_known, observed_residual) =
            replay_reference_ordering_program(region, Some(&observations), Some(analysis.as_ref()));

        // Recording source instructions must not change ordering or either emitted program. The nested read has no
        // public outputs, so it must be recorded based on the residual instructions it emits.
        assert_eq!(known.to_string(), observed_known.to_string());
        assert_eq!(residual.to_string(), observed_residual.to_string());
        assert_eq!(known.instructions().len(), 1);
        assert_eq!(*observations.borrow(), vec![InstructionId::new(region.id(), 0)]);
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation_preserves_validation() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_configuration(None, true, &[]).unwrap();
        let observations = RefCell::new(Vec::new());
        let context = PartialEvaluationContext::new(TracingContext::<TestValue, TestOperation>::new());

        // Invalid input arity is rejected before replay, regardless of observation or reference analysis.
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), None, None),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), Some(&observations), None),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), None, Some(analysis.as_ref())),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), Some(&observations), Some(analysis.as_ref())),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(observations.borrow().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_inline_partitioned_program() {
        // Partition `x - -a` with `a` known. Its single-operation halves can serve as boundary operations;
        // subtraction makes swapping the known coefficient and unknown input observable in the result.
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let difference = builder.add_instruction(SubOperation::new(), Vec::new(), vec![x, negated], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![difference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    context.unknown_input(ArrayType::scalar(DataType::F64), 0),
                ],
                |_| (ArrayOperation::Neg(NegOperation::new()), Vec::new()),
                |_| (ArrayOperation::Sub(SubOperation::new()), Vec::new()),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(-2.0_f64).unwrap())],
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(evaluation.program.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(1)]);
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::scalar(5.0 - (-2.0_f64)).unwrap()]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_partitioned_program_all_known() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true]).unwrap();

        // Preserve the empty residual program's boundary with a region wrapper. It emits no outputs and is removed
        // by finalization because it has no observable effects.
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Array::scalar(2.0).unwrap())],
                |_| (ArrayOperation::Neg(NegOperation::new()), Vec::new()),
                |program| (ArrayOperation::LinearCall(LinearCallOperation::new(0)), vec![program.clone(), program]),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(-2.0_f64).unwrap()));
        let evaluation = context.into_evaluation(outputs).unwrap();
        assert!(evaluation.inputs.is_empty());
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Known(Array::scalar(-2.0_f64).unwrap())]);
        assert!(evaluation.program.instructions().is_empty());
        assert!(evaluation.program.input_ids().is_empty());
        assert!(evaluation.program.output_ids().is_empty());
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]), Ok(vec![Array::scalar(-2.0_f64).unwrap()]));
    }

    #[test]
    fn test_partial_evaluation_context_known_constant() {
        // `known_constant` recovers a known value's staged-constant payload. An eager known value always resolves to a
        // constant, while under a staging known-side context only literal-backed tracers do.
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(context.known_constant(&Array::scalar(5.0).unwrap()), Ok(Array::scalar(5.0).unwrap()));
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());
        assert_eq!(staging_context.known_constant(&literal), Ok(Array::scalar(4.0).unwrap()));
        assert!(matches!(
            staging_context.known_constant(&symbolic),
            Err(ProgramError::MalformedProgram(message))
                if message == "a known value crossing into a nested residual program does not resolve to a constant \
                    in the active known-side context",
        ),);
    }

    #[test]
    fn test_partial_evaluation_context_all_knowns_are_constants() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());

        // A known feeder still occupies an input of the residual program.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(context.all_knowns_are_constants(&PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(Array::scalar(1.0).unwrap())],
            outputs: vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(Array::scalar(2.0).unwrap())
            ],
        }),);

        assert!(!staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(symbolic.clone())],
            outputs: vec![PartialEvaluationOutput::Unknown(0)],
        }),);

        assert!(staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(literal.clone())],
            outputs: vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Known(literal)],
        }),);

        // Symbolic outputs must also reject constant-only reconstruction even when every feeder is constant.
        assert!(!staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program,
            inputs: vec![PartialEvaluationInput::Unknown(0)],
            outputs: vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Known(symbolic)],
        }),);
    }

    #[test]
    fn test_partial_evaluation_context_any_known_is_symbolic() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());

        // `any_known_is_symbolic` is the signal online boundary rules split on. Only a known value that does not
        // resolve to a program constant counts, and so eager knowns and unknowns never do.
        assert!(!context.any_known_is_symbolic(&[PartialEvaluationValue::known(Array::scalar(1.0).unwrap())]));
        assert!(!staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(literal)]));
        assert!(staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(symbolic)]));
        assert!(
            !staging_context
                .any_known_is_symbolic(&[staging_context.unknown_input(ArrayType::scalar(DataType::F64), 0)]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_lift() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        assert_eq!(
            lifted.value().unwrap().materialization(),
            PartialValueMaterialization::Constant { residual_atom: None },
        );
        assert_eq!(lifted.value().unwrap().as_known(), Some(&Array::scalar(2.0).unwrap()));
        let truth = context.lift(Array::scalar(true).unwrap()).unwrap();
        assert_eq!(truth.concretize(), Ok(true));
    }

    #[test]
    fn test_partial_evaluation_context_bind() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        let folded = context.bind(AddOperation::new(), Vec::new(), &[lifted.clone(), lifted.clone()]).unwrap();
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].value().unwrap().as_known(), Some(&Array::scalar(4.0).unwrap()));
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        // A mixed bind retains the known operand as a feeder and stages the multiplication.
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F64), 0));
        let mixed = context.bind(MulOperation::new(), Vec::new(), &[folded[0].clone(), unknown.clone()]).unwrap();
        assert!(mixed[0].value().unwrap().is_unknown());
        let output = mixed[0].value().unwrap().clone();
        drop((lifted, folded, unknown, mixed));
        let evaluation = context.into_evaluation(vec![output]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(4.0).unwrap())],
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = mul %1 %0
            in (%2)
        "}
            .trim_end(),
        );
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(12.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_materialization_error() {
        // A staged reference incorrectly marked as a literal has no constant payload to materialize. Binding a
        // swap retains that error, so later reads with valid operands also fail without emitting parent work.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let reference = outer.input(reference_type);
        let context = PartialEvaluationContext::new(outer.clone());
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(scalar_type, 0));
        let malformed = PartialTracer::new(context.clone(), PartialEvaluationValue::known_constant(reference.clone()));
        let swap = TestOperation::ReferenceSwap(ReferenceSwapOperation::new());
        let poisoned = context.bind(swap, Vec::new(), &[malformed, unknown]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert!(matches!(poisoned[0].value(), Err(ProgramError::MalformedProgram(message))
                if message == "residual materialization required a constant payload for a known value \
                               that is not resolvable to a constant in the active known-side context"),);
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let known = PartialTracer::new(context.clone(), PartialEvaluationValue::known(reference));
        let later = context.bind(read, Vec::new(), &[known]).unwrap();
        assert!(matches!(later[0].value(), Err(ProgramError::MalformedProgram(message))
                if message == "residual materialization required a constant payload for a known value \
                               that is not resolvable to a constant in the active known-side context"),);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_mismatched_builder_error() {
        // A failed bind (in this case, folding an operation whose known inputs belong to two different traces) does
        // not return an error; it poisons its outputs so the infallible operator sugar driving closures never panics.
        // The poison propagates through later binds, resolves `Opaque`, rejects concretizing extractions with the
        // deferred error, and surfaces that original error at the value boundary.
        let outer_a = TestArrayTracingContext::new();
        let outer_b = TestArrayTracingContext::new();
        let context = PartialEvaluationContext::new(outer_a.clone());
        let known_a = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::known(outer_a.input(ArrayType::scalar(DataType::F64))),
        );
        let known_b = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::known(outer_b.input(ArrayType::scalar(DataType::F64))),
        );
        let poisoned = context.bind(AddOperation::new(), Vec::new(), &[known_a.clone(), known_b]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert_eq!(format!("{}", poisoned[0]), "<poison:f64[]>");
        assert_eq!(poisoned[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(context.resolve(&poisoned[0]), ValueResolution::Opaque));
        assert!(matches!(poisoned[0].concretize(), Err(ProgramError::MismatchedProgramBuilders)));

        // Poison propagates from inputs to outputs of later binds, and unwrapping at a boundary reports the original
        // deferred error rather than a generic poison error.
        let propagated = context.bind(MulOperation::new(), Vec::new(), &[known_a, poisoned[0].clone()]).unwrap();
        assert!(matches!(propagated[0].value(), Err(ProgramError::MismatchedProgramBuilders)));
        assert!(matches!(propagated[0].clone().into_value(), Err(ProgramError::MismatchedProgramBuilders)));
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_zero_output_failure() {
        use crate::programs::ReferenceError;

        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let failed_reference = context.lift(TestValue::Reference(frozen)).unwrap();
        let later_reference = context.lift(TestValue::Reference(live.clone())).unwrap();
        let update = context.lift(TestValue::Array(Array::scalar(3.0_f32).unwrap())).unwrap();
        let failed = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[failed_reference, update.clone()]);
        assert!(failed.unwrap().is_empty());

        // Clones retain the same failure and prevent unrelated later mutations, even without a poisoned operand.
        let later = context.clone().bind(ReferenceWriteOperation::new(), Vec::new(), &[later_reference, update]);
        assert!(later.unwrap().is_empty());
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let error = context.into_evaluation(Vec::new()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_discarded_output_failure() {
        use crate::programs::ReferenceError;

        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let failed_reference = context.lift(TestValue::Reference(frozen)).unwrap();
        let poisoned = context.bind(ReferenceReadOperation::new(), Vec::new(), &[failed_reference]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert_eq!(poisoned[0].value().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen),);
        drop(poisoned);

        let later_reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(3.0_f32).unwrap()));
        let error = context
            .fold_or_residualize(ReferenceWriteOperation::new(), Vec::new(), &[later_reference, update])
            .unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let output = PartialEvaluationValue::known(TestValue::Array(Array::scalar(4.0_f32).unwrap()));
        let error = context.into_evaluation(vec![output]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_is_eager() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert!(context.is_eager());
        assert!(!PartialEvaluationContext::new(TestArrayTracingContext::new()).is_eager());
    }

    #[test]
    fn test_partial_evaluation_context_provenance() {
        let parent = TestArrayTracingContext::new();
        let (context, initial) = parent.invoke_with_provenance_scope(ProvenanceScope::new("parent"), || {
            (PartialEvaluationContext::new(parent.clone()), parent.provenance())
        });
        let cloned = context.clone();
        assert_eq!(context.provenance(), initial);
        assert_eq!(cloned.provenance(), initial);
        assert_eq!(parent.provenance(), Provenance::unknown());

        // The residual context snapshots its parent's provenance at construction, then shares its own active state
        // across clones. A local scope changes neither the parent nor the state restored after the scope exits.
        context.invoke_with_provenance_scope(ProvenanceScope::new("residual"), || {
            assert_eq!(context.provenance(), Provenance::scope(ProvenanceScope::new("residual"), initial.clone()));
            assert_eq!(cloned.provenance(), context.provenance());
            assert_eq!(parent.provenance(), Provenance::unknown());
        });
        assert_eq!(context.provenance(), initial);
        assert_eq!(cloned.provenance(), initial);
    }

    #[test]
    fn test_partial_evaluation_context_resolve() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        assert!(
            matches!(context.resolve(&lifted), ValueResolution::Constant(value) if value == Array::scalar(2.0).unwrap())
        );
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F64), 0));
        assert!(matches!(context.resolve(&unknown), ValueResolution::Opaque));
        assert!(matches!(unknown.concretize(), Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from an unknown partial-evaluation value"),);
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity() {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let parent_reference = outer.input(reference_type.clone());
        let expected = outer.reference_identity(&parent_reference).unwrap();
        let context = PartialEvaluationContext::new(outer);
        let known = PartialEvaluationValue::known(parent_reference);
        let original = PartialTracer::new(context.clone(), known.clone());
        assert_eq!(context.reference_identity(&original), Ok(expected));

        // A folded access keeps the known root's parent identity. An unknown reference input belongs to the
        // residual builder's separate namespace.
        let operation = TestOperation::ReferenceRead(ReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) },
        ]));
        let read = context.residualize(operation, Vec::new(), &[known]).unwrap();
        assert_eq!(read[0].r#type().as_ref(), &ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(context.reference_identity(&original), Ok(expected));
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(reference_type, 0));
        assert!(matches!(context.reference_identity(&unknown), Ok(Some(ReferenceIdentity::Staged { .. }))));
        assert_ne!(context.reference_identity(&unknown).unwrap(), expected);

        let value =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F32).into(), 1));
        assert_eq!(context.reference_identity(&value), Ok(None));
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity_for_captured_accesses() {
        let outer = TracingContext::<
            TestCapture,
            ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
        >::new();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let capture = outer.constant(CaptureReference::new(0, reference_type.clone()));
        let expected = outer.reference_identity(&capture).unwrap();
        let context = PartialEvaluationContext::new(outer.clone());
        let captured = PartialEvaluationValue::known_constant(capture);
        let view = context
            .residualize(
                ReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]),
                Vec::new(),
                std::slice::from_ref(&captured),
            )
            .unwrap()
            .remove(0);
        let original = PartialTracer::new(context.clone(), captured);
        let view = PartialTracer::new(context.clone(), view);
        let parent_atom_count = outer.builder().borrow().atoms().len();
        assert_eq!(context.reference_identity(&original), Ok(expected));
        assert_eq!(context.reference_identity(&view), Ok(None));
        assert_eq!(outer.builder().borrow().atoms().len(), parent_atom_count);

        // The residualized access materialized the capture as a residual constant and recorded its parent identity
        // in state that clones share, so a residual variable naming that constant resolves through a clone.
        let PartialValueMaterialization::Constant { residual_atom: Some(residual_atom) } =
            original.value().unwrap().materialization()
        else {
            panic!("residualizing the access must materialize the captured reference as a residual constant");
        };
        let residual = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::variable(reference_type.clone(), residual_atom),
        );
        assert_eq!(context.clone().reference_identity(&residual), Ok(expected));

        let other = context.lift(CaptureReference::new(1, reference_type)).unwrap();
        assert_ne!(context.reference_identity(&other).unwrap(), expected);
        assert_eq!(
            context.reference_identity(&other),
            outer.reference_identity(other.value().unwrap().as_known().unwrap()),
        );
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity_for_unresolved_inherited_capture() {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let outer = TracingContext::<
            TestCapture,
            ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
        >::new();
        let context = PartialEvaluationContext::new(outer.clone());

        // Inherited constants enter the residual builder without being materialized through the parent context.
        // Seed that state directly: no parent identity has been recorded for this symbolic capture.
        let captured = context.builder.borrow_mut().add_constant(CaptureReference::new(0, reference_type.clone()));
        let captured = PartialTracer::new(context.clone(), PartialEvaluationValue::variable(reference_type, captured));
        assert_eq!(context.reference_identity(&captured), Ok(None));
        assert!(outer.builder().borrow().atoms().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_zero() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let zero = context.zero(&ArrayType::scalar(DataType::Boolean)).unwrap();
        assert_eq!(zero.value().unwrap().as_known(), Some(&Array::scalar(false).unwrap()));
        assert_eq!(zero.concretize(), Ok(false));
    }
}
