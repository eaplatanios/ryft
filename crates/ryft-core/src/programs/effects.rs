use std::fmt::Display;

use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::types::Type;

/// Named class of observable effects that an [`Operation`](crate::Operation) can have. Effect classes exist because
/// [`Program`](crate::Program) transforms and backend lowering can have behavior conditional on those classes. For
/// example, XLA lowering threads StableHLO token chains for backend-supported ordered classes to preserve execution
/// order, mirroring [JAX's design](https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html). Ordered state
/// is instead discharged before ordinary XLA lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EffectClass {
    /// Observable runtime assertion whose execution order relative to other
    /// [`OrderedAssertion`](Self::OrderedAssertion) effects determines which failing requirement is reported first.
    /// Operations with this effect must not be eliminated without execution unless their requirement has been proven,
    /// and the relative execution order of retained assertions must be preserved.
    OrderedAssertion,

    /// Observable input/output (e.g., printing) [`EffectClass`] whose execution order relative to other
    /// [`OrderedIo`](Self::OrderedIo) effects is observable (e.g., interleaved printed output). Operations with this
    /// effect must not be folded away or get eliminated, and their relative execution order must be preserved.
    OrderedIo,

    /// Observable input/output (e.g., printing) [`EffectClass`] whose execution order relative to other effects is not
    /// observable. Operations with this effect must not be folded away or get eliminated, but independent unordered-I/O
    /// effects may execute in any order.
    UnorderedIo,

    /// Observable access to mutable state whose execution order relative to other [`OrderedState`](Self::OrderedState)
    /// effects on the same state is observable and must be preserved. This effect _orders_ and does not gate
    /// transforms. Partial evaluation, linearization, and differentiation place stateful operations by the ordered
    /// effect frontier documented in [`partial`](crate::partial) instead of rejecting them, while dead-code elimination
    /// keeps them alive unless the only state behavior of the instruction is a reference allocation that nothing
    /// accesses (refer to [`ReferenceEffect::Allocate`]). Stateful operations must still be either discharged before
    /// stateless lowering or handled by a state-aware backend. Structured reference effects derive this class through
    /// [`Effects::classes`]; an operation lists it explicitly only for opaque state with no structured
    /// reference description, and generic consumers must not infer a particular state representation from the class.
    /// Keeping state distinct from I/O also prevents generic transforms from treating mutation like an external I/O
    /// effect.
    OrderedState,
}

impl EffectClass {
    /// All declared effect classes, in bit order, backing [`EffectClasses`]'s [`IntoIterator`] implementation.
    const ALL: [EffectClass; 4] =
        [EffectClass::OrderedAssertion, EffectClass::OrderedIo, EffectClass::UnorderedIo, EffectClass::OrderedState];

    /// Returns the bit representing this effect class inside an [`EffectClasses`] set.
    const fn bit(self) -> u8 {
        match self {
            EffectClass::OrderedAssertion => 1 << 0,
            EffectClass::OrderedIo => 1 << 1,
            EffectClass::UnorderedIo => 1 << 2,
            EffectClass::OrderedState => 1 << 3,
        }
    }

    /// Returns `true` if the execution order of this effect class relative to other effects of the same class is
    /// observable and must be preserved.
    pub const fn is_ordered(self) -> bool {
        match self {
            EffectClass::OrderedState | EffectClass::OrderedAssertion | EffectClass::OrderedIo => true,
            EffectClass::UnorderedIo => false,
        }
    }
}

/// Set of observable effect classes of an [`Operation`](crate::Operation), describing the behaviors the operation has
/// beyond computing outputs from inputs, such as printing. [`Program`](crate::Program) transforms consult this
/// classification, instead of hardcoding operation lists, before folding, eliminating, or reordering
/// [`Instruction`](crate::Instruction)s. For example:
///
///   - Dead-code elimination ([`Program::simplified`](crate::Program::simplified) and
///     [`Program::into_simplified`](crate::Program::into_simplified)) keeps instructions with observable effects alive
///     even when no program output consumes their results.
///   - [Ordered](Self::is_ordered) effect classes additionally promise that the relative execution order of same-class
///     instructions is preserved. Transforms that would interleave or reorder such instructions with respect to each
///     other must keep them on one side of any split they introduce. For example, XLA lowering threads StableHLO token
///     chains for the ordered classes it supports directly and rejects unresolved ordered state.
///
/// An operation's own classes are derived from its [`Effects`] declaration through [`Effects::classes`]. The
/// classification of a whole [`Program`](crate::Program) is the [`union`](Self::union) of its instructions' classes
/// together with those of their attached computation regions, derived once when regions are sealed and obtained via
/// [`Program::effects`](crate::Program::effects), so that effects remain visible through higher-order boundaries such
/// as loop bodies and compiled function callees without the region-bearing operation restating them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EffectClasses {
    /// Bit set over [`EffectClass::bit`]s.
    bits: u8,
}

impl EffectClasses {
    /// Empty [`EffectClasses`] set.
    pub const NONE: EffectClasses = EffectClasses { bits: 0 };

    /// Returns the [`EffectClasses`] set containing only `effect_class`.
    pub const fn single(effect_class: EffectClass) -> EffectClasses {
        EffectClasses { bits: effect_class.bit() }
    }

    /// Returns the union of this [`EffectClasses`] set and `other`.
    pub const fn union(self, other: EffectClasses) -> EffectClasses {
        EffectClasses { bits: self.bits | other.bits }
    }

    /// Returns `true` if this [`EffectClasses`] set is empty (i.e., if it is equal to [`EffectClasses::NONE`]).
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Returns `true` if this [`EffectClasses`] set contains `effect_class`.
    pub const fn contains(self, effect_class: EffectClass) -> bool {
        self.bits & effect_class.bit() != 0
    }

    /// Returns `true` if this [`EffectClasses`] set contains any class whose execution order is observable.
    pub const fn is_ordered(self) -> bool {
        let mut index = 0;
        while index < EffectClass::ALL.len() {
            let effect_class = EffectClass::ALL[index];
            if effect_class.is_ordered() && self.contains(effect_class) {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Iterator over the effect classes contained in an [`EffectClasses`] set, yielded in declaration order.
pub struct EffectClassesIterator {
    /// [`EffectClasses`] set whose contained effect classes are being iterated over.
    classes: EffectClasses,

    /// Index into [`EffectClass::ALL`] of the next candidate effect class to consider.
    index: usize,
}

impl Iterator for EffectClassesIterator {
    type Item = EffectClass;

    #[inline]
    fn next(&mut self) -> Option<EffectClass> {
        while self.index < EffectClass::ALL.len() {
            let effect_class = EffectClass::ALL[self.index];
            self.index += 1;
            if self.classes.contains(effect_class) {
                return Some(effect_class);
            }
        }
        None
    }
}

impl IntoIterator for EffectClasses {
    type Item = EffectClass;
    type IntoIter = EffectClassesIterator;

    fn into_iter(self) -> EffectClassesIterator {
        EffectClassesIterator { classes: self, index: 0 }
    }
}

/// Represents the type of [`Reference`](crate::Reference) access performed by an [`Operation`](crate::Operation).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceAccessMode {
    /// Reads the referenced value without replacing it.
    Read,

    /// Replaces the selected referenced value without observing its previous contents.
    Write,

    /// Observes the selected referenced value and replaces it with a successor in program order.
    /// [`ReferenceSwapOperation`](crate::ReferenceSwapOperation) remains [`ReferenceAccessMode::ReadWrite`] even
    /// when a caller leaves its old-value result dead: liveness is a use-site fact, not operation semantics.
    ReadWrite,

    /// Combines an update with the current state as an _ordered_ additive accumulation. Accumulation stays distinct
    /// from [`ReferenceAccessMode::Write`] because it is linear in the update operand and therefore transposable,
    /// unlike a replacement. It carries no commutativity or atomicity promise: same-allocation accumulations execute
    /// in program order (floating-point addition cannot generally be reordered while preserving results), and
    /// atomic/commutative accumulation is not supported by this mode.
    Accumulate,

    /// Consumes the allocation. After such an access, the allocation and its entire alias family are invalid.
    /// Consumption is a lifetime event, and not a type of memory-access.
    /// [`ReferenceFreezeOperation`](crate::ReferenceFreezeOperation) is the canonical consuming access operations
    /// that also returns the final value.
    Consume,
}

impl ReferenceAccessMode {
    /// Returns whether this [`ReferenceAccessMode`] consumes the complete reference allocation.
    pub const fn is_consuming(self) -> bool {
        match self {
            Self::Read | Self::Write | Self::ReadWrite | Self::Accumulate => false,
            Self::Consume => true,
        }
    }
}

impl Display for ReferenceAccessMode {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(formatter, "read"),
            Self::Write => write!(formatter, "write"),
            Self::ReadWrite => write!(formatter, "read/write"),
            Self::Accumulate => write!(formatter, "accumulate"),
            Self::Consume => write!(formatter, "consume"),
        }
    }
}

/// [`Reference`](crate::Reference) effect of an [`Operation`](crate::Operation), expressed in the operation's own
/// input/operand and output/result index space. Unlike an [`EffectClass`], a reference effect names the operand or
/// result it targets so that program-level reference analysis can resolve it to a canonical allocation (i.e., an entry
/// input, a capture, or an allocation instruction). The indices are never resource identifiers, so the same declaration
/// is valid for every application of the operation.
///
/// Every reference effect is an occurrence of [`EffectClass::OrderedState`]: [`Effects::classes`] derives that class
/// from the presence of any reference effect, and so reference operations never declare it separately. Aliasing is
/// deliberately _not_ a reference effect, because creating a new handle onto an existing allocation has no observable
/// behavior. Refer to [`ReferenceAlias`] for the alias declaration that lives beside reference effects.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceEffect {
    /// The output at `output_index` allocates a fresh reference and defines a new canonical allocation. Allocation is
    /// an effect because an allocation must never be duplicated or merged with another (i.e., each application mints a
    /// distinct mutable cell), but it is _unobservable when unused_ meaning that an allocation that nothing accesses
    /// can be eliminated together with its dead alias family. An allocation whose creation or abandonment has an
    /// observable failure, synchronization, or external lifetime consequence must additionally declare the
    /// corresponding explicit [`EffectClass`], which keeps it alive.
    Allocate {
        /// Index of the [`Operation`](crate::Operation) output defining the new allocation.
        output_index: usize,
    },

    /// The reference input at `input_index` is accessed with `mode`. Every access is observable when unused, including
    /// a read, because an access can wait for pending backend completion, reconcile pending state, and report
    /// poisoning, a consumed reference, or a mutex poisoning; eliminating an unused access would remove an observable
    /// synchronization point or error.
    Access {
        /// Index of the [`Operation`](crate::Operation) input being accessed.
        input_index: usize,

        /// [`ReferenceAccessMode`] of the access.
        mode: ReferenceAccessMode,
    },
}

/// Declaration that the [`Reference`](crate::Reference)-valued output of an [`Operation`](crate::Operation) at
/// `output_index` is an alias of the canonical allocation of its reference-valued input at `input_index`. An alias
/// carries exactly one `input_index` because every reference operand must resolve to exactly one canonical allocation,
/// so multi-source aliases (e.g., a hypothetical `select_reference(a, b)`) are structurally unrepresentable rather than
/// merely rejected. [`ReferenceAliasKind`] distinguishes an identity-preserving edge from an operation-owned view edge.
/// Generic allocation analysis needs only that marker, and the value family's discharge policy obtains and validates
/// the exact view metadata through the operation family's view-operation contract.
///
/// Aliases are declared beside [`ReferenceEffect`]s in [`Effects`] because reference effects cannot be resolved to
/// allocations without them, but an alias is reference identity/dataflow rather than an effect (i.e., it contributes
/// no [`EffectClass`], and an operation that only aliases like a static view, for example, remains pure).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceAlias {
    /// Index of the [`Operation`](crate::Operation) output producing the alias.
    output_index: usize,

    /// [`Operation`](crate::Operation) input index whose canonical allocation is preserved.
    input_index: usize,

    /// [`ReferenceAliasKind`] specifying whether this alias preserves the exact handle or adds an
    /// [`Operation`](crate::Operation)-owned view mapping.
    kind: ReferenceAliasKind,
}

impl ReferenceAlias {
    /// Creates a new [`ReferenceAlias`].
    pub const fn new(output_index: usize, input_index: usize, kind: ReferenceAliasKind) -> Self {
        Self { output_index, input_index, kind }
    }

    /// Returns the index of the [`Operation`](crate::Operation) output producing the alias.
    pub const fn output_index(self) -> usize {
        self.output_index
    }

    /// Returns the [`Operation`](crate::Operation) input index whose canonical allocation is preserved.
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the [`ReferenceAliasKind`] of this [`ReferenceAlias`].
    pub const fn kind(self) -> ReferenceAliasKind {
        self.kind
    }
}

/// Kind of an allocation-preserving [`ReferenceAlias`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceAliasKind {
    /// The alias preserves the input handle's exact referent type and mapping.
    Identity,

    /// The alias selects a view of the input handle's allocation, and the aliasing operation itself carries the
    /// metadata that maps the new handle's coordinates onto that allocation. The generic program layer records only
    /// that the output aliases the input's allocation; interpreting and validating the operation-owned metadata is the
    /// job of the value family's reference discharge policy, which obtains it through the operation family's reference
    /// view contract.
    ///
    /// For example, [`ReferenceSliceOperation`](crate::ReferenceSliceOperation) declares
    /// `ReferenceAlias::new(0, 0, ReferenceAliasKind::View)` specifying that its result is a handle onto the same
    /// allocation whose referent is the sliced window, the slice axes live on the operation, and the array discharge
    /// policy reads those axes to materialize or reconstruct the selected coordinates during discharge.
    View,
}

/// Complete operation-local effect declaration of an [`Operation`](crate::Operation) that contains its explicitly
/// declared effect classes, its [`ReferenceEffect`]s, and its [`ReferenceAlias`]es. This is the single authoritative
/// declaration from which the aggregate [`EffectClasses`] consumed by transforms and lowering, the reference facts
/// consumed by [`ReferenceAnalysis`](crate::ReferenceAnalysis) and discharge, and the retention decision of dead-code
/// elimination are all derived.
///
/// The declaration is _intrinsic_ to the operation and expressed in its own input/operand and output/result index
/// space. [`Region`](crate::Region)-bearing operations (e.g., loops and conditionals) declare only what they do
/// themselves, which is usually nothing, even when their nested programs are effectful or touch references (region
/// sealing aggregates nested effects into region metadata, and reference analysis recurses into attached regions
/// rather than trusting per-instruction declarations alone). How values cross an attached region's boundary is a
/// separate contract, declared through the region-boundary hooks of [`Operation`](crate::Operation) (e.g.,
/// [`Operation::reference_output_identity_input`](crate::Operation::reference_output_identity_input) and
/// [`Operation::output_region_provenance`](crate::Operation::output_region_provenance)), which are positional
/// constraints that analysis applies only to reference-typed positions; this declaration never restates them.
///
/// Declarations are stored in canonical order regardless of the order in which they were constructed. Specifically,
/// accesses are sorted by input index, then allocations are sorted by output index, and finally aliases are sorted by
/// output index. Two declarations with the same facts therefore compare equal, and iteration order is deterministic
/// for diagnostics and analysis records.
///
/// # Declared Effect Classes
///
/// The declared classes are the [`EffectClass`]es the operation author lists directly (e.g., assertions, I/O, and
/// _opaque_ ordered state that has no structured reference description). [`Self::classes`] additionally derives
/// [`EffectClass::OrderedState`] from the presence of any [`ReferenceEffect`], so an operation with reference effects
/// must list `OrderedState` explicitly only for genuinely opaque state, never to classify its reference effects. That
/// convention cannot be enforced here, because a redundant class is indistinguishable from an operation that has both
/// a structured access and opaque state. [`EffectsSummary::has_explicit_ordered_state`] retains the distinction across
/// nested computations, so partial evaluation and rematerialization cannot mistake this additional state for a confined
/// reference lifecycle.
///
/// # Examples
///
/// Reference operations declare no explicit effect classes. Each of the six primitives declares one [`ReferenceEffect`]
/// and no aliases:
///
///   | Operation                          | Reference Effect                              |
///   | ---------------------------------- | --------------------------------------------- |
///   | `reference_new(x) -> r`            | `Allocate { output_index: 0 }`                |
///   | `reference_read(r) -> x`           | `Access { input_index: 0, mode: Read }`       |
///   | `reference_write(r, x) -> ()`      | `Access { input_index: 0, mode: Write }`      |
///   | `reference_swap(r, x) -> old`      | `Access { input_index: 0, mode: ReadWrite }`  |
///   | `reference_add_update(r, x) -> ()` | `Access { input_index: 0, mode: Accumulate }` |
///   | `reference_freeze(r) -> x`         | `Access { input_index: 0, mode: Consume }`    |
///
/// The two view operations, `reference_index(r, axis, index) -> view` and `reference_slice(r, axes) -> view`, declare
/// only `ReferenceAlias { output_index: 0, input_index: 0, kind: View }`. By contrast, `print(x) -> ()` declares only
/// the explicit class [`EffectClass::OrderedIo`], with no reference effects or aliases.
///
/// Their derived [`classes`](Self::classes) are `{OrderedState}` for the six primitive reference operations, `NONE` for
/// the two views, and `{OrderedIo}` for the `print` operation. Of these, only the unused `reference_new` and the views
/// are eliminated by dead-code elimination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Effects {
    /// Aggregate [`EffectClass`]es, observability, and explicit state derived once from the declaration.
    summary: EffectsSummary,

    /// [`ReferenceEffect`]s in canonical order. Single Static Assignment (SSA) value (i.e., non-reference) inputs and
    /// outputs are omitted from this list.
    reference_effects: Vec<ReferenceEffect>,

    /// [`ReferenceAlias`]es in canonical order. Single Static Assignment (SSA) value (i.e., non-reference) outputs are
    /// omitted from this list.
    reference_aliases: Vec<ReferenceAlias>,
}

impl Effects {
    /// Creates a new [`Effects`] declaration from its components, canonicalizing the order of the provided
    /// `reference_effects` and `reference_aliases`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] when one input index receives two accesses or one output index
    /// receives two classifications (i.e., two allocations, two aliases, or an allocation and an alias). Each position
    /// must appear at most once so reference analysis can trust that every output is either a fresh allocation or
    /// exactly one alias. Index ranges cannot be checked here because the declaration carries no arity information;
    /// the [`ProgramBuilder`](crate::ProgramBuilder) validates them against each instruction's actual operand/result
    /// arity.
    pub fn new(
        declared: EffectClasses,
        mut reference_effects: Vec<ReferenceEffect>,
        mut reference_aliases: Vec<ReferenceAlias>,
    ) -> Result<Self, ProgramError> {
        // Accesses precede allocations so that the canonical order reads inputs first and then outputs.
        reference_effects.sort_by_key(|effect| match effect {
            ReferenceEffect::Access { input_index, .. } => (0, *input_index),
            ReferenceEffect::Allocate { output_index } => (1, *output_index),
        });
        reference_aliases.sort_by_key(|alias| alias.output_index);

        // Canonical ordering places duplicate input accesses and output allocations next to one another.
        for pair in reference_effects.windows(2) {
            match (&pair[0], &pair[1]) {
                (
                    ReferenceEffect::Access { input_index: previous, .. },
                    ReferenceEffect::Access { input_index, .. },
                ) if previous == input_index => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("input {input_index} received two reference accesses"),
                    });
                }
                (ReferenceEffect::Allocate { output_index: previous }, ReferenceEffect::Allocate { output_index })
                    if previous == output_index =>
                {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("output {output_index} received two reference classifications"),
                    });
                }
                _ => {}
            }
        }
        for (index, alias) in reference_aliases.iter().enumerate() {
            let output_index = alias.output_index;
            if (index > 0 && reference_aliases[index - 1].output_index == output_index)
                || reference_effects.iter().any(|effect| {
                    matches!(
                        effect,
                        ReferenceEffect::Allocate { output_index: allocated } if *allocated == output_index,
                    )
                })
            {
                return Err(ProgramError::InvalidArgument {
                    message: format!("output {output_index} received two reference classifications"),
                });
            }
        }

        let has_access = reference_effects.iter().any(|effect| matches!(effect, ReferenceEffect::Access { .. }));
        let classes = if reference_effects.is_empty() {
            declared
        } else {
            declared.union(EffectClasses::single(EffectClass::OrderedState))
        };
        let summary = EffectsSummary {
            classes,
            has_observable_effects_when_unused: !declared.is_empty() || has_access,
            has_explicit_ordered_state: declared.contains(EffectClass::OrderedState),
        };
        Ok(Self { summary, reference_effects, reference_aliases })
    }

    /// Returns the shared empty [`Effects`] declaration of pure [`Operation`](crate::Operation)s that neither
    /// create, alias, nor access [`Reference`](crate::Reference)s.
    #[inline]
    pub fn empty() -> &'static Self {
        &EMPTY_EFFECTS
    }

    /// Creates a new [`Effects`] declaration that consists only of the provided directly declared effect classes,
    /// which is the declaration of assertion, I/O, and opaque-state operations that neither create, alias, nor access
    /// references.
    #[inline]
    pub fn explicit(classes: EffectClasses) -> Self {
        Self::new(classes, Vec::new(), Vec::new()).unwrap()
    }

    /// Returns the [`EffectsSummary`] derived from this declaration.
    #[inline]
    pub fn summary(&self) -> EffectsSummary {
        self.summary
    }

    /// Returns the aggregate [`EffectClasses`] of the declaring [`Operation`](crate::Operation) that consist of
    /// its directly declared classes combined with [`EffectClass::OrderedState`] when it declares at least one
    /// [`ReferenceEffect`]. Aliases contribute no effect class.
    #[inline]
    pub fn classes(&self) -> EffectClasses {
        self.summary.classes
    }

    /// Returns `true` if the aggregate [`classes`](Self::classes) are [`EffectClasses::NONE`].
    #[inline]
    pub fn is_pure(&self) -> bool {
        self.summary.classes.is_empty()
    }

    /// Returns `true` if the [`Operation`](crate::Operation) declares `effect_class` directly, rather than it being
    /// derived from reference effects. [`EffectsSummary::has_explicit_ordered_state`] preserves this distinction for
    /// [`EffectClass::OrderedState`] across nested computations.
    #[inline]
    pub fn declares(&self, effect_class: EffectClass) -> bool {
        match effect_class {
            // Only ordered state can be synthesized by reference declarations.
            EffectClass::OrderedState => self.summary.has_explicit_ordered_state,
            _ => self.summary.classes.contains(effect_class),
        }
    }

    /// Returns the [`ReferenceEffect`]s of the declaring [`Operation`](crate::Operation) in canonical order
    /// (i.e., accesses sorted by input index come first, followed by allocations sorted by output index).
    #[inline]
    pub fn reference_effects(&self) -> &[ReferenceEffect] {
        self.reference_effects.as_slice()
    }

    /// Returns the [`ReferenceAlias`]es of the declaring [`Operation`](crate::Operation) in canonical order
    /// (i.e., sorted by output index).
    #[inline]
    pub fn reference_aliases(&self) -> &[ReferenceAlias] {
        self.reference_aliases.as_slice()
    }

    /// Returns `true` if this [`Effects`] declaration names at least one reference access. Allocations, aliases,
    /// reference-typed boundaries, and reference-typed constants are not considered accesses.
    #[inline]
    pub fn has_accesses(&self) -> bool {
        self.reference_effects.iter().any(|effect| matches!(effect, ReferenceEffect::Access { .. }))
    }

    /// Returns the `(input_index, mode)` pairs of the declared reference accesses of this [`Effects`] declaration,
    /// in ascending input index order.
    #[inline]
    pub fn accesses(&self) -> impl Iterator<Item = (usize, ReferenceAccessMode)> {
        self.reference_effects.iter().filter_map(|effect| match effect {
            ReferenceEffect::Access { input_index, mode } => Some((*input_index, *mode)),
            ReferenceEffect::Allocate { .. } => None,
        })
    }

    /// Returns the output positions at which the declaring [`Operation`](crate::Operation) allocates a fresh
    /// reference, in ascending output index order.
    #[inline]
    pub fn allocation_output_indices(&self) -> impl Iterator<Item = usize> {
        self.reference_effects.iter().filter_map(|effect| match effect {
            ReferenceEffect::Allocate { output_index } => Some(*output_index),
            ReferenceEffect::Access { .. } => None,
        })
    }

    /// Returns `true` if this declaration names at least one [`ReferenceEffect`] or [`ReferenceAlias`] (i.e., if the
    /// declaring [`Operation`](crate::Operation) creates, aliases, or accesses references). This is intentionally not
    /// called `is_empty` because an operation with ordered I/O and no reference declarations is far from effect-free.
    #[inline]
    pub fn has_reference_declarations(&self) -> bool {
        !self.reference_effects.is_empty() || !self.reference_aliases.is_empty()
    }

    /// Validates this [`Effects`] declaration against one [`Operation`](crate::Operation) application. For validation
    /// to succeed, every named input and output position must exist in the application, and every named position must
    /// be reference-typed, because reference effects and aliases describe reference allocations and a declaration on a
    /// non-reference operand or result could never be resolved by reference analysis. Opaque state on non-reference
    /// values is declared through an explicit [`EffectClass::OrderedState`] instead.
    ///
    /// # Parameters
    ///
    ///   - `operation_name`: Name of the operation whose declaration is being validated, used for diagnostic purposes.
    ///   - `input_types`: Types of the inputs/operands of the application.
    ///   - `output_types`: Types of the outputs/results inferred for the application.
    pub(crate) fn validate_application<T: Type>(
        &self,
        operation_name: &str,
        input_types: &[T],
        output_types: &[T],
    ) -> Result<(), ProgramError> {
        let validate_input = |input_index: usize, role: &str| match input_types.get(input_index) {
            Some(r#type) if r#type.is_reference() => Ok(()),
            Some(r#type) => Err(ProgramError::MalformedProgram(format!(
                "operation `{operation_name}` names {role} input {input_index} but it has non-reference type `{type}`",
            ))),
            None => Err(ProgramError::MalformedProgram(format!(
                "operation `{}` names {} input {} but the application input count is {}",
                operation_name,
                role,
                input_index,
                input_types.len(),
            ))),
        };

        let validate_output = |output_index: usize| match output_types.get(output_index) {
            Some(r#type) if r#type.is_reference() => Ok(()),
            Some(r#type) => Err(ProgramError::MalformedProgram(format!(
                "operation `{operation_name}` classifies output {output_index} but it has non-reference type `{type}`",
            ))),
            None => Err(ProgramError::MalformedProgram(format!(
                "operation `{}` classifies output {} but the application output count is {}",
                operation_name,
                output_index,
                output_types.len(),
            ))),
        };

        for effect in &self.reference_effects {
            match effect {
                ReferenceEffect::Access { input_index, .. } => validate_input(*input_index, "an accessed")?,
                ReferenceEffect::Allocate { output_index } => validate_output(*output_index)?,
            }
        }

        for alias in &self.reference_aliases {
            validate_output(alias.output_index)?;
            validate_input(alias.input_index, "an aliased")?;
        }

        Ok(())
    }
}

// Shared empty declaration returned by `Effects::empty` so that the `Operation` trait default can hand out a borrow
// without allocating (`Vec::new` is `const`, so this static needs no lazy initialization).
static EMPTY_EFFECTS: Effects =
    Effects { summary: EffectsSummary::PURE, reference_effects: Vec::new(), reference_aliases: Vec::new() };

/// Aggregate summary of [`Effects`] that survives union across [`Instruction`](crate::Instruction)s and
/// [`Region`](crate::Region)s without index translation. It records the aggregate [`EffectClasses`], whether unused
/// results permit elimination, and whether an executable instruction directly declares [`EffectClass::OrderedState`].
/// The latter distinction cannot be recovered from the classes once reference effects derive the same class.
///
/// The observability information reflects the runtime contract for references: a read operation may synchronize with
/// pending backend work or report a reference-state failure, so every access is observable even when unused, while an
/// unused allocation is not. Operations declare [`Effects`], while instruction and region queries derive and
/// aggregate this summary rather than requiring operation authors to construct it directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EffectsSummary {
    /// Aggregate [`EffectClasses`].
    classes: EffectClasses,

    /// Whether an application with no live output still has an observable consequence and must be retained by
    /// dead-code elimination. Explicit nonempty [`EffectClasses`] and reference accesses set this flag, while
    /// allocations and aliases do not.
    has_observable_effects_when_unused: bool,

    /// Whether an operation directly declares [`EffectClass::OrderedState`], independently of its reference effects.
    has_explicit_ordered_state: bool,
}

impl EffectsSummary {
    /// [`EffectsSummary`] of a pure [`Operation`](crate::Operation) application (i.e., no effect classes and nothing
    /// observable when the [`Operation`](crate::Operation) application result is unused).
    pub const PURE: EffectsSummary = EffectsSummary {
        classes: EffectClasses::NONE,
        has_observable_effects_when_unused: false,
        has_explicit_ordered_state: false,
    };

    /// Returns the aggregate [`EffectClasses`] of this [`EffectsSummary`].
    pub const fn classes(self) -> EffectClasses {
        self.classes
    }

    /// Returns whether an application with no live output still has an observable consequence and must be retained by
    /// dead-code elimination.
    pub const fn has_observable_effects_when_unused(self) -> bool {
        self.has_observable_effects_when_unused
    }

    /// Returns whether any executable instruction directly declares [`EffectClass::OrderedState`], rather than deriving
    /// it solely from [`ReferenceEffect`]s. A direct declaration describes state that reference analysis cannot assign
    /// to the declared reference roots, even when the instruction also accesses those roots. Partial evaluation
    /// therefore orders it against every root, and rematerialization cannot replay it as part of a confined reference
    /// lifecycle. Checking [`classes`](Self::classes) alone cannot distinguish these cases.
    pub const fn has_explicit_ordered_state(self) -> bool {
        self.has_explicit_ordered_state
    }

    /// Returns the union of this [`EffectsSummary`] with `other`, combining [`EffectClasses`] and retaining either flag
    /// when either side sets it. An enclosing operation cannot suppress an observable nested effect or an explicit
    /// ordered-state declaration.
    pub const fn union(self, other: EffectsSummary) -> EffectsSummary {
        EffectsSummary {
            classes: self.classes.union(other.classes),
            has_observable_effects_when_unused: self.has_observable_effects_when_unused
                || other.has_observable_effects_when_unused,
            has_explicit_ordered_state: self.has_explicit_ordered_state || other.has_explicit_ordered_state,
        }
    }
}

/// Occurrence of an [`EffectClass`] intrinsically carried by an [`Instruction`](crate::Instruction)
/// in a [`Region`](crate::Region).
pub struct EffectClassOccurrence<'o, O> {
    /// Location of the [`Instruction`](crate::Instruction) corresponding to this occurrence in the source
    /// [`Region`](crate::Region) arena.
    instruction: InstructionId,

    /// [`EffectClass`]-carrying [`Operation`](crate::Operation).
    operation: &'o O,
}

impl<'o, O> EffectClassOccurrence<'o, O> {
    /// Creates a new [`EffectClassOccurrence`].
    #[inline]
    pub(crate) fn new(instruction: InstructionId, operation: &'o O) -> Self {
        Self { instruction, operation }
    }

    /// Returns the location of the [`Instruction`](crate::Instruction) corresponding to this occurrence in the source
    /// [`Region`](crate::Region) arena.
    #[inline]
    pub fn instruction(&self) -> InstructionId {
        self.instruction
    }

    /// Returns the [`EffectClass`]-carrying [`Operation`](crate::Operation).
    #[inline]
    pub fn operation(&self) -> &'o O {
        self.operation
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayType, DataType};
    use crate::programs::references::ReferenceType;
    use crate::programs::regions::RegionId;

    use super::*;

    #[test]
    fn test_effect_classes() {
        // The empty set contains nothing, reports no ordering, and iterates over nothing.
        assert!(EffectClasses::NONE.is_empty());
        assert!(!EffectClasses::NONE.contains(EffectClass::OrderedAssertion));
        assert!(!EffectClasses::NONE.contains(EffectClass::OrderedIo));
        assert!(!EffectClasses::NONE.contains(EffectClass::UnorderedIo));
        assert!(!EffectClasses::NONE.contains(EffectClass::OrderedState));
        assert!(!EffectClasses::NONE.is_ordered());
        assert_eq!(EffectClasses::NONE.into_iter().collect::<Vec<_>>(), Vec::<EffectClass>::new());

        // A singleton set contains only its effect, and every ordered class reports observable ordering.
        let assertion = EffectClasses::single(EffectClass::OrderedAssertion);
        let ordered_io = EffectClasses::single(EffectClass::OrderedIo);
        let unordered = EffectClasses::single(EffectClass::UnorderedIo);
        let ordered_state = EffectClasses::single(EffectClass::OrderedState);
        assert!(!assertion.is_empty());
        assert!(assertion.contains(EffectClass::OrderedAssertion));
        assert!(!assertion.contains(EffectClass::OrderedIo));
        assert!(assertion.is_ordered());
        assert!(!ordered_io.is_empty());
        assert!(!ordered_io.contains(EffectClass::OrderedAssertion));
        assert!(ordered_io.contains(EffectClass::OrderedIo));
        assert!(!ordered_io.contains(EffectClass::UnorderedIo));
        assert!(ordered_io.is_ordered());
        assert!(!unordered.is_empty());
        assert!(!unordered.is_ordered());
        assert!(!ordered_state.is_empty());
        assert!(ordered_state.contains(EffectClass::OrderedState));
        assert!(ordered_state.is_ordered());
        assert_eq!(assertion.into_iter().collect::<Vec<_>>(), vec![EffectClass::OrderedAssertion]);
        assert_eq!(ordered_io.into_iter().collect::<Vec<_>>(), vec![EffectClass::OrderedIo]);
        assert_eq!(unordered.into_iter().collect::<Vec<_>>(), vec![EffectClass::UnorderedIo]);
        assert_eq!(ordered_state.into_iter().collect::<Vec<_>>(), vec![EffectClass::OrderedState]);

        // Union is commutative and idempotent, `NONE` is its identity element, and the combined set contains every
        // class and iterates in declaration order.
        let all = assertion.union(ordered_io).union(unordered).union(ordered_state);
        assert_eq!(all, unordered.union(ordered_io).union(assertion).union(ordered_state));
        assert_eq!(all.union(all), all);
        assert_eq!(all.union(EffectClasses::NONE), all);
        assert_eq!(EffectClasses::NONE.union(assertion), assertion);
        assert!(!all.is_empty());
        assert!(all.contains(EffectClass::OrderedAssertion));
        assert!(all.contains(EffectClass::OrderedIo));
        assert!(all.contains(EffectClass::UnorderedIo));
        assert!(all.contains(EffectClass::OrderedState));
        assert!(all.is_ordered());
        assert_eq!(
            all.into_iter().collect::<Vec<_>>(),
            vec![
                EffectClass::OrderedAssertion,
                EffectClass::OrderedIo,
                EffectClass::UnorderedIo,
                EffectClass::OrderedState
            ],
        );

        // Equality distinguishes distinct sets, self-equality holds for rebuilt sets, and hashing supports map lookups.
        assert_eq!(assertion, EffectClasses::single(EffectClass::OrderedAssertion));
        assert_ne!(assertion, ordered_io);
        assert_ne!(ordered_io, unordered);
        assert_ne!(ordered_io, all);
        assert_ne!(all, EffectClasses::NONE);
        let lookup = HashMap::from([(assertion, "assertion"), (ordered_io, "ordered I/O"), (all, "all")]);
        assert_eq!(lookup.get(&EffectClasses::single(EffectClass::OrderedAssertion)), Some(&"assertion"));
        assert_eq!(lookup.get(&EffectClasses::single(EffectClass::OrderedIo)), Some(&"ordered I/O"));
        assert_eq!(lookup.get(&unordered.union(ordered_io).union(assertion).union(ordered_state)), Some(&"all"));
        assert_eq!(lookup.get(&unordered), None);
    }

    #[test]
    fn test_reference_access_mode() {
        let cases = [
            (ReferenceAccessMode::Read, "read", false),
            (ReferenceAccessMode::Write, "write", false),
            (ReferenceAccessMode::ReadWrite, "read/write", false),
            (ReferenceAccessMode::Accumulate, "accumulate", false),
            (ReferenceAccessMode::Consume, "consume", true),
        ];
        for (mode, display, is_consuming) in cases {
            assert_eq!(mode.to_string(), display);
            assert_eq!(mode.is_consuming(), is_consuming);
        }
    }

    #[test]
    fn test_reference_alias() {
        let alias = ReferenceAlias::new(2, 1, ReferenceAliasKind::View);
        assert_eq!(alias.output_index(), 2);
        assert_eq!(alias.input_index(), 1);
        assert_eq!(alias.kind(), ReferenceAliasKind::View);
        assert_eq!(alias, ReferenceAlias::new(2, 1, ReferenceAliasKind::View));
        assert_ne!(alias, ReferenceAlias::new(2, 1, ReferenceAliasKind::Identity));
        assert_ne!(alias, ReferenceAlias::new(1, 2, ReferenceAliasKind::View));
    }

    #[test]
    fn test_effects() {
        // The shared empty declaration is pure, declares nothing, and equals a freshly constructed empty declaration.
        let empty = Effects::empty();
        assert_eq!(empty.classes(), EffectClasses::NONE);
        assert!(empty.is_pure());
        assert!(!empty.declares(EffectClass::OrderedState));
        assert_eq!(empty.reference_effects(), &[]);
        assert_eq!(empty.reference_aliases(), &[]);
        assert_eq!(empty.accesses().collect::<Vec<_>>(), Vec::<(usize, ReferenceAccessMode)>::new());
        assert!(!empty.has_accesses());
        assert_eq!(empty.allocation_output_indices().collect::<Vec<_>>(), Vec::<usize>::new());
        assert!(!empty.has_reference_declarations());
        assert_eq!(empty.summary(), EffectsSummary::PURE);
        assert_eq!(empty, &Effects::new(EffectClasses::NONE, vec![], vec![]).unwrap());
        assert_eq!(empty, &Effects::explicit(EffectClasses::NONE));

        // Directly declared classes are also part of the aggregate, and any nonempty directly declared class set is
        // observable when unused.
        let io = Effects::explicit(EffectClasses::single(EffectClass::OrderedIo));
        assert_eq!(io.classes(), EffectClasses::single(EffectClass::OrderedIo));
        assert!(!io.is_pure());
        assert!(io.declares(EffectClass::OrderedIo));
        assert!(!io.declares(EffectClass::OrderedState));
        assert!(!io.has_reference_declarations());
        assert!(io.summary().has_observable_effects_when_unused());

        // An allocation derives `OrderedState` without the author listing it, but is not observable when unused.
        let allocation =
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], vec![]).unwrap();
        assert_eq!(allocation.classes(), EffectClasses::single(EffectClass::OrderedState));
        assert!(!allocation.is_pure());
        assert!(!allocation.declares(EffectClass::OrderedState));
        assert_eq!(allocation.reference_effects(), &[ReferenceEffect::Allocate { output_index: 0 }]);
        assert_eq!(allocation.allocation_output_indices().collect::<Vec<_>>(), vec![0]);
        assert_eq!(allocation.accesses().collect::<Vec<_>>(), Vec::<(usize, ReferenceAccessMode)>::new());
        assert!(!allocation.has_accesses());
        assert!(allocation.has_reference_declarations());
        assert_eq!(allocation.summary().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert!(!allocation.summary().has_observable_effects_when_unused());

        // An access, including a read, derives `OrderedState` and is observable when unused.
        let read = Effects::new(
            EffectClasses::NONE,
            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            vec![],
        )
        .unwrap();
        assert_eq!(read.classes(), EffectClasses::single(EffectClass::OrderedState));
        assert!(!read.declares(EffectClass::OrderedState));
        assert_eq!(read.accesses().collect::<Vec<_>>(), vec![(0, ReferenceAccessMode::Read)]);
        assert!(read.has_accesses());

        assert_eq!(read.allocation_output_indices().collect::<Vec<_>>(), Vec::<usize>::new());
        assert!(read.summary().has_observable_effects_when_unused());

        // An alias contributes no class: a view operation remains pure and discardable while still declaring
        // references.

        let view = Effects::new(EffectClasses::NONE, vec![], vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)])
            .unwrap();
        assert_eq!(view.classes(), EffectClasses::NONE);
        assert!(view.is_pure());
        assert_eq!(view.reference_effects(), &[]);
        assert_eq!(view.reference_aliases(), &[ReferenceAlias::new(0, 0, ReferenceAliasKind::View)]);
        assert!(view.has_reference_declarations());
        assert_eq!(view.summary(), EffectsSummary::PURE);

        // A mixed declaration composes conservatively: directly declared opaque state stays distinguishable from the
        // derived class, and one observable component retains the whole application.
        let mixed = Effects::new(
            EffectClasses::single(EffectClass::OrderedState)
                .union(EffectClasses::single(EffectClass::OrderedAssertion)),
            vec![ReferenceEffect::Allocate { output_index: 1 }],
            vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity)],
        )
        .unwrap();
        assert_eq!(
            mixed.classes(),
            EffectClasses::single(EffectClass::OrderedState)
                .union(EffectClasses::single(EffectClass::OrderedAssertion)),
        );
        assert!(mixed.declares(EffectClass::OrderedState));
        assert!(mixed.declares(EffectClass::OrderedAssertion));
        assert!(!mixed.declares(EffectClass::OrderedIo));
        assert!(mixed.summary().has_observable_effects_when_unused());

        // Equality distinguishes distinct declarations and clones compare equal.
        assert_ne!(allocation, read);
        assert_ne!(io, Effects::explicit(EffectClasses::single(EffectClass::UnorderedIo)));
        assert_eq!(mixed.clone(), mixed);
    }

    #[test]
    fn test_effects_canonical_order() {
        // Author order is not significant: accesses sort by input index and precede allocations, which sort by output
        // index, and aliases sort by output index. Distinct output aliases may share the same source input.
        let effects = Effects::new(
            EffectClasses::NONE,
            vec![
                ReferenceEffect::Allocate { output_index: 3 },
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                ReferenceEffect::Allocate { output_index: 1 },
                ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
            ],
            vec![
                ReferenceAlias::new(2, 0, ReferenceAliasKind::View),
                ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
            ],
        )
        .unwrap();
        assert_eq!(
            effects.reference_effects(),
            &[
                ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                ReferenceEffect::Allocate { output_index: 1 },
                ReferenceEffect::Allocate { output_index: 3 },
            ],
        );
        assert_eq!(
            effects.reference_aliases(),
            &[
                ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
                ReferenceAlias::new(2, 0, ReferenceAliasKind::View)
            ],
        );
        assert_eq!(
            effects.accesses().collect::<Vec<_>>(),
            vec![(0, ReferenceAccessMode::Read), (2, ReferenceAccessMode::Write)],
        );
        assert_eq!(effects.allocation_output_indices().collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(
            effects,
            Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 1 },
                    ReferenceEffect::Allocate { output_index: 3 },
                ],
                vec![
                    ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
                    ReferenceAlias::new(2, 0, ReferenceAliasKind::View),
                ],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_effects_rejects_two_accesses_for_one_input() {
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                ],
                vec![],
            ),
            Err(ProgramError::InvalidArgument { message }) if message == "input 2 received two reference accesses",
        ));
    }

    #[test]
    fn test_effects_rejects_two_allocations_for_one_output() {
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Allocate { output_index: 3 },
                    ReferenceEffect::Allocate { output_index: 1 },
                    ReferenceEffect::Allocate { output_index: 3 },
                ],
                vec![],
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "output 3 received two reference classifications",
        ));
    }

    #[test]
    fn test_effects_rejects_two_aliases_for_one_output() {
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![],
                vec![
                    ReferenceAlias::new(2, 0, ReferenceAliasKind::Identity),
                    ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
                    ReferenceAlias::new(2, 1, ReferenceAliasKind::View),
                ],
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "output 2 received two reference classifications",
        ));
    }

    #[test]
    fn test_effects_rejects_allocation_and_alias_for_one_output() {
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Allocate { output_index: 0 }],
                vec![
                    ReferenceAlias::new(2, 0, ReferenceAliasKind::View),
                    ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
                ],
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "output 0 received two reference classifications",
        ));
    }

    #[test]
    fn test_effects_rejection_precedence() {
        // Duplicate accesses precede output conflicts, regardless of their indices or declaration order.
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![
                    ReferenceEffect::Allocate { output_index: 0 },
                    ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Allocate { output_index: 0 },
                    ReferenceEffect::Access { input_index: 3, mode: ReferenceAccessMode::Write },
                ],
                vec![],
            ),
            Err(ProgramError::InvalidArgument { message }) if message == "input 3 received two reference accesses",
        ));

        // Duplicate allocations precede alias conflicts, even when the aliases have smaller output indices.
        assert!(matches!(
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Allocate { output_index: 7 }, ReferenceEffect::Allocate { output_index: 7 }],
                vec![
                    ReferenceAlias::new(0, 0, ReferenceAliasKind::Identity),
                    ReferenceAlias::new(0, 1, ReferenceAliasKind::View),
                ],
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "output 7 received two reference classifications",
        ));
    }

    #[test]
    fn test_effects_validate_application() {
        let array = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let reference = ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let read = Effects::new(
            EffectClasses::NONE,
            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            vec![],
        )
        .unwrap();
        let alias =
            Effects::new(EffectClasses::NONE, vec![], vec![ReferenceAlias::new(0, 1, ReferenceAliasKind::Identity)])
                .unwrap();
        let allocation =
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], vec![]).unwrap();

        // Well-typed declarations on existing positions are accepted, and so is the empty declaration on anything.
        assert_eq!(read.validate_application("test.read", &[reference.clone()], &[array.clone()]), Ok(()));
        assert_eq!(
            alias.validate_application("test.alias", &[array.clone(), reference.clone()], &[reference.clone()]),
            Ok(())
        );
        assert_eq!(
            allocation.validate_application("test.reference_new", &[array.clone()], &[reference.clone()]),
            Ok(())
        );
        assert_eq!(Effects::empty().validate_application("test.add", &[array.clone()], &[array.clone()]), Ok(()));

        // Out-of-range positions are rejected with the application's arity.
        assert_eq!(
            read.validate_application::<ArrayIrType>("test.read", &[], &[]),
            Err(ProgramError::MalformedProgram(
                "operation `test.read` names an accessed input 0 but the application input count is 0".to_string(),
            )),
        );
        assert_eq!(
            alias.validate_application("test.alias", &[reference.clone()], &[reference.clone()]),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` names an aliased input 1 but the application input count is 1".to_string(),
            )),
        );
        assert_eq!(
            alias.validate_application("test.alias", &[array.clone(), reference.clone()], &[]),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` classifies output 0 but the application output count is 0".to_string(),
            )),
        );
        assert_eq!(
            allocation.validate_application("test.reference_new", &[array.clone()], &[]),
            Err(ProgramError::MalformedProgram(
                "operation `test.reference_new` classifies output 0 but the application output count is 0".to_string(),
            )),
        );

        // Non-reference endpoints are rejected for accesses, aliased inputs, alias outputs, and allocations alike.
        assert_eq!(
            read.validate_application("test.read", &[array.clone()], &[array.clone()]),
            Err(ProgramError::MalformedProgram(
                "operation `test.read` names an accessed input 0 but it has non-reference type `f32[]`".to_string(),
            )),
        );
        assert_eq!(
            alias.validate_application("test.alias", &[reference.clone(), array.clone()], &[reference.clone()]),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` names an aliased input 1 but it has non-reference type `f32[]`".to_string(),
            )),
        );
        assert_eq!(
            alias.validate_application("test.alias", &[array.clone(), reference.clone()], &[array.clone()]),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` classifies output 0 but it has non-reference type `f32[]`".to_string(),
            )),
        );
        assert_eq!(
            allocation.validate_application("test.reference_new", &[array.clone()], &[array]),
            Err(ProgramError::MalformedProgram(
                "operation `test.reference_new` classifies output 0 but it has non-reference type `f32[]`".to_string(),
            )),
        );
    }

    #[test]
    fn test_effects_summary() {
        let allocation =
            Effects::new(EffectClasses::NONE, vec![ReferenceEffect::Allocate { output_index: 0 }], vec![]).unwrap();
        let read = Effects::new(
            EffectClasses::NONE,
            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            vec![],
        )
        .unwrap();
        let io = Effects::explicit(EffectClasses::single(EffectClass::OrderedIo));

        // The pure summary is the identity element of `union`.
        assert_eq!(EffectsSummary::PURE.classes(), EffectClasses::NONE);
        assert!(!EffectsSummary::PURE.has_observable_effects_when_unused());
        assert_eq!(EffectsSummary::PURE.union(EffectsSummary::PURE), EffectsSummary::PURE);
        assert_eq!(EffectsSummary::PURE.union(read.summary()), read.summary());
        assert_eq!(io.summary().union(EffectsSummary::PURE), io.summary());

        // Union combines classes and retains the observable-when-unused bit if either side sets it, so an
        // allocation-only outer summary cannot suppress a nested access or I/O effect, while two allocation-only
        // summaries stay discardable.
        let allocation_and_read: EffectsSummary = allocation.summary().union(read.summary());
        assert_eq!(allocation_and_read.classes(), EffectClasses::single(EffectClass::OrderedState));
        assert!(allocation_and_read.has_observable_effects_when_unused());
        assert_eq!(allocation_and_read, read.summary().union(allocation.summary()));
        let allocation_and_io = allocation.summary().union(io.summary());
        assert_eq!(
            allocation_and_io.classes(),
            EffectClasses::single(EffectClass::OrderedState).union(EffectClasses::single(EffectClass::OrderedIo)),
        );
        assert!(allocation_and_io.has_observable_effects_when_unused());
        assert_eq!(allocation.summary().union(allocation.summary()), allocation.summary());
        assert!(!allocation.summary().union(allocation.summary()).has_observable_effects_when_unused());
    }

    #[test]
    fn test_effects_summary_has_explicit_ordered_state() {
        let read = Effects::new(
            EffectClasses::NONE,
            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            Vec::new(),
        )
        .unwrap();
        let mixed = Effects::new(
            EffectClasses::single(EffectClass::OrderedState),
            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
            Vec::new(),
        )
        .unwrap();
        let explicit = Effects::explicit(EffectClasses::single(EffectClass::OrderedState));

        // Both declarations have the same aggregate classes and observability, but only the directly declared
        // state prevents transforms from attributing every effect to known reference roots.
        assert_eq!(read.summary().classes(), mixed.summary().classes());
        assert_eq!(read.summary().classes(), explicit.summary().classes());
        assert!(read.summary().has_observable_effects_when_unused());
        assert!(mixed.summary().has_observable_effects_when_unused());
        assert!(!EffectsSummary::PURE.has_explicit_ordered_state());
        assert!(!read.summary().has_explicit_ordered_state());
        assert!(explicit.summary().has_explicit_ordered_state());
        assert!(mixed.summary().has_explicit_ordered_state());
        assert!(read.summary().union(mixed.summary()).has_explicit_ordered_state());
        assert!(mixed.summary().union(read.summary()).has_explicit_ordered_state());
    }

    #[test]
    fn test_effect_class_occurrence() {
        let operation = "effectful";
        let instruction = InstructionId::new(RegionId::new(2), 3);
        let occurrence = EffectClassOccurrence::new(instruction, &operation);
        assert_eq!(occurrence.instruction(), instruction);
        assert_eq!(occurrence.operation(), &operation);
    }
}
