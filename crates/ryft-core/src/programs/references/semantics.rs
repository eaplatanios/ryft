//! Structural reference types, operation-local reference semantics, entry-boundary reference sources, and the
//! trace-time liveness state derived from them.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::fmt::Display;

use ryft_macros::Parameter;
use serde::Serialize;

use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::identities::{TypeIdentityPosition, TypeIdentityRenaming};
use crate::programs::operations::Operation;
use crate::programs::types::{Type, TypeError, TypeRefinements};

/// Meaning of one reference input access performed by an operation.
///
/// The modes form a deliberately small vocabulary mirroring JAX's `ReadEffect`/`WriteEffect`/`AccumEffect` reference
/// effects, plus consumption as a lifetime event. There is intentionally no `ReadWrite` mode: `Write` asserts that
/// the access writes, not that no read occurs (see [`ReferenceAccessMode::Write`]), which matches JAX classifying
/// `swap` as a plain write.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceAccessMode {
    /// Reads the current state without replacing it.
    Read,

    /// Replaces the current state. `Write` is an over-approximation floor: it asserts that the access *writes*, not
    /// that no read occurs. An operation may still observe the previous state through its results (e.g.,
    /// `reference_swap` returns the old value), so generic analyses must treat a `Write` conservatively as possibly
    /// reading prior state through its results — in particular, any future dead-store elimination on a state chain
    /// must not remove an earlier write based on this mode alone. Whether a specific write actually reads is an
    /// operation-specific question (e.g., kernel lowering of `reference_swap` knows its own old-value result and can
    /// emit a plain store when that result is dead); this descriptor deliberately carries no access-to-result
    /// mapping until a generic analysis genuinely needs one.
    Write,

    /// Combines an update with the current state as an *ordered* additive accumulation. Accumulation stays distinct
    /// from [`Write`](ReferenceAccessMode::Write) because it is linear in the update operand and therefore
    /// transposable, unlike a replacement. It carries no commutativity or atomicity promise: same-root accumulations
    /// execute in program order (floating-point addition cannot generally be reordered while preserving results),
    /// and atomic/commutative accumulation is a separate future semantics.
    Accumulate,

    /// Consumes the root: after this access the root and its entire alias family are invalid. Consumption is a
    /// lifetime event, not a memory-access flavor — `freeze` is the consuming access that also returns the final
    /// value, and a future `free_reference` would consume without producing a result.
    Consume,
}

impl Display for ReferenceAccessMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(formatter, "read"),
            Self::Write => write!(formatter, "write"),
            Self::Accumulate => write!(formatter, "accumulate"),
            Self::Consume => write!(formatter, "consume"),
        }
    }
}

/// Kind of one root-preserving reference alias edge.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceAliasKind {
    /// The alias preserves the input handle's exact referent type and mapping.
    Identity,

    /// The alias adds operation-owned view metadata validated by the value family's owning analysis overlay.
    View,
}

/// Reference classification of one operation output: either a fresh canonical root or an alias of one input's root.
///
/// The two cases are mutually exclusive by construction, so an output can never be declared as both a new root and
/// an alias. [`Alias`](ReferenceOutputSemantics::Alias) carries exactly one `input_index` on purpose: every reference
/// operand must resolve to exactly one canonical root, so multi-source aliases (e.g., a hypothetical
/// `select_reference(a, b)`) are structurally unrepresentable rather than merely rejected. [`ReferenceAliasKind`]
/// distinguishes an identity-preserving edge from an operation-owned view edge. Generic root analysis needs only that
/// marker; the value family's owning overlay obtains and validates its exact view metadata from the operation family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceOutputSemantics {
    /// The output allocates a fresh resource and defines a new canonical root.
    NewRoot {
        /// Operation output index defining the new root.
        output_index: usize,
    },

    /// The output aliases the canonical root of one reference input.
    Alias {
        /// Operation output index producing the alias.
        output_index: usize,

        /// Operation input index whose canonical root is preserved.
        input_index: usize,

        /// Whether this edge preserves the exact handle or adds an operation-owned view mapping.
        kind: ReferenceAliasKind,
    },
}

impl ReferenceOutputSemantics {
    /// Returns the operation output index this classification applies to.
    #[inline]
    pub const fn output_index(self) -> usize {
        match self {
            Self::NewRoot { output_index } | Self::Alias { output_index, .. } => output_index,
        }
    }
}

/// Reference access performed through one operation input.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceInputAccess {
    /// Operation input index being accessed.
    input_index: usize,

    /// Semantic mode of the access.
    mode: ReferenceAccessMode,
}

impl ReferenceInputAccess {
    /// Creates an access of `mode` through `input_index`.
    #[inline]
    pub const fn new(input_index: usize, mode: ReferenceAccessMode) -> Self {
        Self { input_index, mode }
    }

    /// Returns the operation input index being accessed.
    #[inline]
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the semantic mode of this access.
    #[inline]
    pub const fn mode(self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Operation-local reference semantics: output root/alias classification plus input accesses, expressed in
/// operand/result index space.
///
/// All indices are *operation-local operand and result positions*, never resource identifiers. Program-level
/// analysis later resolves them to canonical roots (an entry input, a capture, or an allocation instruction), so the
/// descriptor intentionally contains no runtime resource IDs. The empty default describes operations that neither
/// create, alias, nor access references.
///
/// # Examples
///
/// The array-reference vocabulary declares the following semantics:
///
/// ```text
/// new_reference(x) -> r
///     outputs  = [NewRoot { output_index: 0 }]
///     accesses = []
///
/// reference_read(r) -> x
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Read }]
///
/// reference_swap(r, x) -> old
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Write }]
///
/// reference_add_update(r, x) -> ()
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Accumulate }]
///
/// freeze_reference(r) -> x
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Consume }]
///
/// reference_index(r, axis, index) -> view
/// reference_slice(r, axes) -> view
///     outputs  = [Alias { output_index: 0, input_index: 0, kind: View }]
///     accesses = []
/// ```
///
/// Note that `reference_swap` declares `Write` even though its result observes the previous value: `Write` does not
/// assert the absence of a read (refer to [`ReferenceAccessMode::Write`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceOperationSemantics {
    /// Reference classifications of operation outputs.
    outputs: Vec<ReferenceOutputSemantics>,

    /// State accesses through reference inputs.
    accesses: Vec<ReferenceInputAccess>,
}

// Shared empty descriptor returned by `ReferenceOperationSemantics::empty` so that the `Operation` trait default can
// hand out a borrow without allocating (`Vec::new` is `const`, so this static needs no lazy initialization).
static EMPTY_REFERENCE_OPERATION_SEMANTICS: ReferenceOperationSemantics =
    ReferenceOperationSemantics { outputs: Vec::new(), accesses: Vec::new() };

impl ReferenceOperationSemantics {
    /// Creates a reference semantics descriptor from its ordered components.
    ///
    /// # Panics
    ///
    /// Panics when one output index receives two classifications or one input index receives two accesses. These are
    /// operation-author contract violations: the documented mutual exclusivity of [`ReferenceOutputSemantics`] holds
    /// only if each operand/result position appears at most once, and program-level analysis trusts that invariant.
    /// Index ranges cannot be checked here because the descriptor carries no arity information; program-level
    /// reference analysis validates them against each instruction's actual operand/result arity.
    pub fn new(outputs: Vec<ReferenceOutputSemantics>, accesses: Vec<ReferenceInputAccess>) -> Self {
        for (index, output) in outputs.iter().enumerate() {
            let output_index = output.output_index();
            assert!(
                outputs[..index].iter().all(|previous| previous.output_index() != output_index),
                "output {output_index} received two reference classifications",
            );
        }
        for (index, access) in accesses.iter().enumerate() {
            let input_index = access.input_index();
            assert!(
                accesses[..index].iter().all(|previous| previous.input_index() != input_index),
                "input {input_index} received two reference accesses",
            );
        }
        Self { outputs, accesses }
    }

    /// Returns the shared empty descriptor used by operations that neither create, alias, nor access references.
    #[inline]
    pub fn empty() -> &'static Self {
        &EMPTY_REFERENCE_OPERATION_SEMANTICS
    }

    /// Returns whether this descriptor declares no reference outputs and no reference accesses.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty() && self.accesses.is_empty()
    }

    /// Returns output reference classifications in deterministic operation-defined order.
    #[inline]
    pub fn outputs(&self) -> &[ReferenceOutputSemantics] {
        self.outputs.as_slice()
    }

    /// Returns the output positions at which this operation allocates a fresh reference root, in deterministic
    /// operation-defined order.
    #[inline]
    pub fn new_root_output_indices(&self) -> impl Iterator<Item = usize> {
        self.outputs.iter().filter_map(|output| match output {
            ReferenceOutputSemantics::NewRoot { output_index } => Some(*output_index),
            ReferenceOutputSemantics::Alias { .. } => None,
        })
    }

    /// Returns reference accesses in deterministic operation-defined order.
    #[inline]
    pub fn accesses(&self) -> &[ReferenceInputAccess] {
        self.accesses.as_slice()
    }
}

/// Reference alias topology and consumption state of one region under construction, which is what lets a misuse of
/// a reference fail at the append that performs it — for a traced program, the staging call — rather than at
/// discharge.
///
/// This is the construction-time rung of the reference prevention ladder. The eager runtime already invalidates a
/// complete alias family when a holder is frozen, and reports a later access as
/// [`ReferenceError::Frozen`](crate::programs::ReferenceError::Frozen); and discharge reports what its own
/// environment observes when a program built outside tracing is rewritten. Between them sat a gap: a program under
/// construction could record a read of a frozen reference and only surface it much later, from a diagnostic that no
/// longer named the call that caused it. One
/// [`ProgramBuilder`](crate::ProgramBuilder) constructs exactly one region — non-entry regions enter it only in
/// sealed form — so this state is per-region by construction, and it is keyed by [`AtomId`], which is the
/// identity every clone of one staged [`Tracer`](crate::Tracer) shares.
///
/// The legality of one append depends on what every earlier append did, so this state is the incrementally
/// maintained fold of those appends: [`validate`](Self::validate) consults it before an instruction is accepted, and
/// [`record`](Self::record) folds an accepted instruction in, keeping the per-append cost proportional to the
/// instruction rather than to the region built so far. The fold covers *checked* appends only:
/// [`ProgramBuilder::add_instruction_unchecked`](crate::ProgramBuilder::add_instruction_unchecked) deliberately
/// bypasses both calls, so this state describes what the checked appends of the region have established rather than
/// the ground truth of the region, and a program assembled through the bypass is left to the eager runtime and
/// discharge rungs.
///
/// Consumption is a whole-family event, so an access is resolved to its alias-family root before the state is
/// consulted. Two facts are therefore tracked: which root each derived handle belongs to, and which operation
/// consumed each root.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReferenceLifetimes {
    /// Name of the consuming operation, per consumed alias-family root.
    consumed: BTreeMap<AtomId, &'static str>,

    /// Alias-family root of each derived reference atom, together with whether the chain reaching it narrows the
    /// referent. Narrowing is transitive: an identity edge onto a view still denotes part of a root, so the flag is
    /// carried forward rather than re-read from the last edge.
    aliases: BTreeMap<AtomId, ResolvedReferenceAlias>,
}

impl ReferenceLifetimes {
    /// Rejects an application that accesses a reference whose alias family an earlier instruction already consumed,
    /// or that consumes a handle whose view narrows the root the consumption would invalidate.
    ///
    /// Out-of-range access indices are ignored here rather than reported, because operand arity belongs to type
    /// inference and reference descriptor well-formedness to the discharge environment; this check owns liveness
    /// alone. Consumption of a non-narrowing handle is likewise accepted regardless of where its root was allocated:
    /// partial discharge deliberately accepts a program that consumes a surviving external root, so rejecting one
    /// here would forbid a capability the transform supports.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation being staged, consulted for its declared reference semantics and its name.
    ///   - `inputs`: Operand atoms of the application, in operation-defined order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] naming both the accessing operation and the operation that
    /// consumed the alias family, or naming the operation that attempted to consume a narrowing view.
    pub(crate) fn validate<O: Operation>(&self, operation: &O, inputs: &[AtomId]) -> Result<(), ProgramError> {
        if self.consumed.is_empty() && self.aliases.is_empty() {
            return Ok(());
        }
        let name = operation.name();
        let semantics = operation.reference_semantics();
        for access in semantics.accesses() {
            let Some(atom) = inputs.get(access.input_index()) else {
                continue;
            };
            let edge = self.aliases.get(atom);
            if access.mode() == ReferenceAccessMode::Consume && edge.is_some_and(|edge| edge.narrows) {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` consumes a derived reference view, but consumption invalidates the whole alias \
                     family; consume the root handle instead",
                )));
            }
            let root = edge.map_or(*atom, |edge| edge.root);
            if let Some(consumer) = self.consumed.get(&root) {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` {}s a reference whose alias family `{consumer}` already consumed",
                    access.mode(),
                )));
            }
        }
        Ok(())
    }

    /// Records the consumptions and the alias edges one accepted application performs.
    ///
    /// Alias edges come from two independent declarations, because a reference reaches an output in two ways. An
    /// access primitive or a view declares [`ReferenceOutputSemantics::Alias`], while a region-carrying operation
    /// that carries a reference through its boundary declares the forwarding through
    /// [`Operation::reference_output_identity_input`](crate::Operation::reference_output_identity_input) instead.
    /// Both make the output part of the operand's alias family, so both are recorded; a declaration of the first
    /// kind wins where an operation makes both, because it additionally states whether the edge narrows.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation being staged, consulted for the reference semantics it declares, for its
    ///     identity-forwarded reference outputs, and for the name recorded against every root it consumes.
    ///   - `inputs`: Operand atoms of the application, in operation-defined order.
    ///   - `outputs`: Result atoms of the application, in operation-defined order.
    pub(crate) fn record<O: Operation>(&mut self, operation: &O, inputs: &[AtomId], outputs: &[AtomId]) {
        let semantics = operation.reference_semantics();
        for access in semantics.accesses() {
            if access.mode() == ReferenceAccessMode::Consume
                && let Some(atom) = inputs.get(access.input_index())
            {
                let root = self.root(*atom);
                self.consumed.insert(root, operation.name());
            }
        }
        for (output_index, output_atom) in outputs.iter().copied().enumerate() {
            let Some(input_index) = operation.reference_output_identity_input(output_index) else {
                continue;
            };
            if let Some(input_atom) = inputs.get(input_index) {
                self.alias(output_atom, *input_atom, false);
            }
        }
        for output in semantics.outputs() {
            if let ReferenceOutputSemantics::Alias { output_index, input_index, kind } = *output
                && let (Some(output_atom), Some(input_atom)) = (outputs.get(output_index), inputs.get(input_index))
            {
                self.alias(*output_atom, *input_atom, kind == ReferenceAliasKind::View);
            }
        }
    }

    /// Records one alias edge from `output` onto `input`, resolving it to the family root immediately so that later
    /// lookups are a single map access rather than a walk.
    ///
    /// # Parameters
    ///
    ///   - `output`: Derived reference atom the edge produces.
    ///   - `input`: Reference atom the edge is composed onto.
    ///   - `narrows`: Whether this edge itself narrows the referent.
    fn alias(&mut self, output: AtomId, input: AtomId, narrows: bool) {
        let source = self.aliases.get(&input).copied();
        let edge = ResolvedReferenceAlias {
            root: source.map_or(input, |source| source.root),
            narrows: narrows || source.is_some_and(|source| source.narrows),
        };
        self.aliases.insert(output, edge);
    }

    /// Returns the alias-family root of one reference atom, which is the atom itself when it is not a derived handle.
    fn root(&self, atom: AtomId) -> AtomId {
        self.aliases.get(&atom).map_or(atom, |edge| edge.root)
    }
}

/// Resolved alias-family membership of one derived reference atom.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResolvedReferenceAlias {
    /// Alias-family root this atom belongs to.
    root: AtomId,

    /// Whether the chain from the root to this atom narrows the referent, which is what forbids consuming it.
    narrows: bool,
}

/// Invocation source of one external reference root.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    /// Capture lifted into the entry boundary before public arguments.
    Capture {
        /// Zero-based capture position in the lifted capture prefix.
        index: usize,
    },

    /// Public reference argument after the lifted capture prefix.
    PublicInput {
        /// Zero-based public input position, excluding lifted captures.
        index: usize,
    },
}

impl ReferenceSource {
    /// Returns the entry-boundary source of one flat input position, splitting the boundary at the lifted capture
    /// prefix.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Flat entry input position.
    ///   - `capture_count`: Number of leading flat inputs that originated in the program's capture table.
    #[inline]
    pub const fn from_input_index(input_index: usize, capture_count: usize) -> Self {
        if input_index < capture_count {
            Self::Capture { index: input_index }
        } else {
            Self::PublicInput { index: input_index - capture_count }
        }
    }
}

impl Display for ReferenceSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::PublicInput { index } => write!(formatter, "public input {index}"),
        }
    }
}

/// Structural [`Type`] of a reference to a value whose type is `T`.
///
/// A reference type contains only referent metadata. Runtime resource identity belongs to
/// [`Reference`](crate::programs::Reference) and therefore does not affect structural equality, hashing, or
/// retained-program specialization. Reference compatibility is exact: a reference cannot implicitly broadcast or
/// promote its storage, while refinement and identity handling delegate to the referent type. Exactness deliberately
/// spans the referent's optional layout, sharding, and memory metadata as well: the external-state mutation ABI
/// requires exact physical referent compatibility, so a metadata-tolerant relation would overpromise. Revisit this if
/// a compatibility consumer ever needs the tolerant reading.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceType<T: Type> {
    /// Structural type of the referenced value.
    referent: T,
}

impl<T: Type> ReferenceType<T> {
    /// Creates a reference type for `referent`.
    #[inline]
    pub fn new(referent: T) -> Self {
        Self { referent }
    }

    /// Returns the structural type of the referenced value.
    #[inline]
    pub fn referent(&self) -> &T {
        &self.referent
    }
}

impl<T: Type> Display for ReferenceType<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ref<{}>", self.referent)
    }
}

impl<T: Type> Type for ReferenceType<T> {
    type Identity = T::Identity;
    type Refinements = ReferenceTypeRefinements<T>;

    #[inline]
    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.referent.identities()
    }

    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
        let declared = declared.iter().map(|r#type| r#type.referent.clone()).collect::<Vec<_>>();
        let actual = actual.iter().map(|r#type| r#type.referent.clone()).collect::<Vec<_>>();
        T::derive_identity_renaming(&declared, &actual)
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        Ok(Self::new(self.referent.rename_identities(renaming)?))
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        self.referent.is_refined_by(&other.referent)
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        false
    }

    #[inline]
    fn is_complex(&self) -> bool {
        false
    }

    #[inline]
    fn is_reference(&self) -> bool {
        true
    }
}

/// Cross-occurrence refinements established for a complete [`ReferenceType`] signature.
#[derive(Clone, Debug)]
pub struct ReferenceTypeRefinements<T: Type> {
    /// Referent refinement state shared across every reference in the signature.
    referents: T::Refinements,
}

impl<T: Type> Default for ReferenceTypeRefinements<T> {
    #[inline]
    fn default() -> Self {
        Self { referents: T::Refinements::default() }
    }
}

impl<T: Type> TypeRefinements<ReferenceType<T>> for ReferenceTypeRefinements<T> {
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ReferenceType<T>>,
        A::Item: Borrow<ReferenceType<T>>,
    {
        // Collecting the items is a shallow move; the referents themselves are delegated by borrow (`&T` satisfies
        // the `Borrow<T>` item bound) so no referent is ever cloned on this type-inference path.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let declared = declared.iter().map(|r#type| &r#type.borrow().referent);
        let actual = actual.iter().map(|r#type| &r#type.borrow().referent);
        Ok(Self { referents: T::Refinements::establish(declared, actual)? })
    }

    fn validate<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[T::Identity],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ReferenceType<T>>,
        A::Item: Borrow<ReferenceType<T>>,
    {
        // Refer to the borrow-based delegation note in `establish` above.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let declared = declared.iter().map(|r#type| &r#type.borrow().referent);
        let actual = actual.iter().map(|r#type| &r#type.borrow().referent);
        self.referents.validate(declared, actual, closed_identities)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::{Borrow, Cow};
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::parameters::Parameter;
    use crate::programs::identities::TypeIdentity;
    use crate::programs::operations::Operation;
    use crate::programs::regions::RegionInterface;

    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    struct TestIdentity(u8);

    impl Display for TestIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "identity<{}>", self.0)
        }
    }

    impl TypeIdentity for TestIdentity {
        fn fresh(&self) -> Self {
            Self(self.0.wrapping_add(128))
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum TestType {
        Dynamic(TestIdentity),
        Static(u8),
    }

    impl Display for TestType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Dynamic(identity) => write!(formatter, "dynamic<{identity}>"),
                Self::Static(value) => write!(formatter, "static<{value}>"),
            }
        }
    }

    impl Parameter for TestType {}

    #[derive(Clone, Debug, Default)]
    struct TestTypeRefinements {
        values: Vec<(TestIdentity, u8)>,
    }

    impl TestTypeRefinements {
        fn observe(&mut self, declared: &TestType, actual: &TestType) -> Result<(), TypeError> {
            match (declared, actual) {
                (TestType::Dynamic(identity), TestType::Static(value)) => {
                    if let Some((_, established)) = self.values.iter().find(|(candidate, _)| candidate == identity) {
                        if established != value {
                            return Err(TypeError::invalid(format!(
                                "identity `{identity}` was refined to both {established} and {value}",
                            )));
                        }
                    } else {
                        self.values.push((*identity, *value));
                    }
                    Ok(())
                }
                (TestType::Dynamic(_), TestType::Dynamic(_)) | (TestType::Static(_), TestType::Static(_))
                    if declared.is_refined_by(actual) =>
                {
                    Ok(())
                }
                _ => Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}"))),
            }
        }
    }

    impl TypeRefinements<TestType> for TestTypeRefinements {
        fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
        where
            D::IntoIter: ExactSizeIterator,
            A::IntoIter: ExactSizeIterator,
            D::Item: Borrow<TestType>,
            A::Item: Borrow<TestType>,
        {
            let declared = declared.into_iter();
            let actual = actual.into_iter();
            if declared.len() != actual.len() {
                return Err(TypeError::invalid(format!(
                    "declared type count {} does not match actual type count {}",
                    declared.len(),
                    actual.len(),
                )));
            }
            let mut refinements = Self::default();
            for (declared, actual) in declared.zip(actual) {
                refinements.observe(declared.borrow(), actual.borrow())?;
            }
            Ok(refinements)
        }

        fn validate<D: IntoIterator, A: IntoIterator>(
            &self,
            declared: D,
            actual: A,
            _closed_identities: &[TestIdentity],
        ) -> Result<(), TypeError>
        where
            D::IntoIter: ExactSizeIterator,
            A::IntoIter: ExactSizeIterator,
            D::Item: Borrow<TestType>,
            A::Item: Borrow<TestType>,
        {
            let declared = declared.into_iter();
            let actual = actual.into_iter();
            if declared.len() != actual.len() {
                return Err(TypeError::invalid(format!(
                    "declared type count {} does not match actual type count {}",
                    declared.len(),
                    actual.len(),
                )));
            }
            let mut refinements = self.clone();
            for (declared, actual) in declared.zip(actual) {
                refinements.observe(declared.borrow(), actual.borrow())?;
            }
            Ok(())
        }
    }

    impl Type for TestType {
        type Identity = TestIdentity;
        type Refinements = TestTypeRefinements;

        fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
            match self {
                Self::Dynamic(identity) => Some((TypeIdentityPosition::Definition, identity)),
                Self::Static(_) => None,
            }
            .into_iter()
        }

        fn derive_identity_renaming(
            declared: &[Self],
            actual: &[Self],
        ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
            Self::Refinements::establish(declared, actual)?;
            let mut renaming = TypeIdentityRenaming::new();
            for (declared, actual) in declared.iter().zip(actual) {
                if let (Self::Dynamic(declared), Self::Dynamic(actual)) = (declared, actual) {
                    renaming.insert(*declared, *actual)?;
                }
            }
            Ok(renaming)
        }

        fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Dynamic(identity) => Self::Dynamic(renaming.rename(identity)),
                Self::Static(value) => Self::Static(*value),
            })
        }

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            matches!(self, Self::Dynamic(_)) || self == other
        }

        fn is_scalar(&self) -> bool {
            false
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_reference_type_delegates_identity_and_refinement_without_implicit_compatibility() {
        let declared = TestIdentity(0);
        let actual = TestIdentity(1);
        let declared_type = ReferenceType::new(TestType::Dynamic(declared));
        let actual_type = ReferenceType::new(TestType::Dynamic(actual));
        let renaming = ReferenceType::derive_identity_renaming(
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
        )
        .unwrap();
        assert_eq!(renaming.rename(&declared), actual);

        let static_two = ReferenceType::new(TestType::Static(2));
        let static_three = ReferenceType::new(TestType::Static(3));
        assert!(declared_type.is_refined_by(&static_two));
        assert!(!declared_type.is_compatible_with(&static_two));
        assert!(!static_two.is_compatible_with(&static_three));
        assert!(static_two.is_reference());
        assert!(!static_two.is_scalar());
        assert!(!static_two.is_complex());
        assert_eq!(static_two.to_string(), "ref<static<2>>");
        assert_eq!(format!("{static_two:?}"), format!("ReferenceType {{ referent: {:?} }}", static_two.referent()));
        let refinements = ReferenceTypeRefinements::establish(
            [declared_type.clone(), declared_type.clone()],
            [static_two.clone(), static_two.clone()],
        )
        .unwrap();
        assert_eq!(refinements.validate([declared_type.clone()], [static_two.clone()], &[]), Ok(()));
        let error = ReferenceTypeRefinements::establish(
            [ReferenceType::new(TestType::Dynamic(declared)), ReferenceType::new(TestType::Dynamic(declared))],
            [static_two, static_three],
        )
        .unwrap_err();
        assert_eq!(error, TypeError::invalid("identity `identity<0>` was refined to both 2 and 3"));
    }

    #[test]
    fn test_reference_semantics_describes_an_alias_without_resource_identity() {
        #[derive(Clone)]
        struct TestAliasingOperation;

        impl Operation for TestAliasingOperation {
            type Type = ReferenceType<TestType>;

            fn name(&self) -> &'static str {
                "test_alias"
            }

            fn infer_output_types(
                &self,
                input_types: &[ReferenceType<TestType>],
                _region_interfaces: &[RegionInterface<ReferenceType<TestType>>],
            ) -> Result<Vec<ReferenceType<TestType>>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                Cow::Owned(ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::Identity,
                    }],
                    Vec::new(),
                ))
            }
        }

        let semantics = TestAliasingOperation.reference_semantics();
        assert!(semantics.accesses().is_empty());
        assert!(!semantics.is_empty());
        assert_eq!(
            semantics.outputs(),
            &[ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(ReferenceOperationSemantics::empty(), &ReferenceOperationSemantics::default());
        assert!(ReferenceOperationSemantics::empty().is_empty());

        let boxed: Box<TestAliasingOperation> = Box::new(TestAliasingOperation);
        assert_eq!(boxed.reference_semantics(), TestAliasingOperation.reference_semantics());
    }

    #[test]
    #[should_panic(expected = "output 0 received two reference classifications")]
    fn test_reference_semantics_rejects_two_classifications_for_one_output() {
        ReferenceOperationSemantics::new(
            vec![
                ReferenceOutputSemantics::NewRoot { output_index: 0 },
                ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity },
            ],
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "input 0 received two reference accesses")]
    fn test_reference_semantics_rejects_two_accesses_for_one_input() {
        ReferenceOperationSemantics::new(
            Vec::new(),
            vec![
                ReferenceInputAccess::new(0, ReferenceAccessMode::Read),
                ReferenceInputAccess::new(0, ReferenceAccessMode::Write),
            ],
        );
    }

    #[test]
    fn test_reference_lifetimes() {
        // One operation stands in for every reference primitive: it carries the semantics under test plus an
        // optional identity-forwarded output, which is how a region-carrying operation declares that it hands a
        // reference back out of its boundary. The state derives everything it records from the operation itself, so
        // a fixture cannot describe one contract and be recorded under another.
        #[derive(Clone)]
        struct TestReferenceOperation {
            name: &'static str,
            semantics: ReferenceOperationSemantics,
            forwarded: Option<(usize, usize)>,
        }

        impl Operation for TestReferenceOperation {
            type Type = ReferenceType<TestType>;

            fn name(&self) -> &'static str {
                self.name
            }

            fn infer_output_types(
                &self,
                input_types: &[ReferenceType<TestType>],
                _region_interfaces: &[RegionInterface<ReferenceType<TestType>>],
            ) -> Result<Vec<ReferenceType<TestType>>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                Cow::Borrowed(&self.semantics)
            }

            fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
                self.forwarded.and_then(|(output, input)| (output == output_index).then_some(input))
            }
        }

        let access = |name, mode| TestReferenceOperation {
            name,
            semantics: ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, mode)]),
            forwarded: None,
        };
        let alias = |name, kind| TestReferenceOperation {
            name,
            semantics: ReferenceOperationSemantics::new(
                vec![ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind }],
                Vec::new(),
            ),
            forwarded: None,
        };
        let allocation = TestReferenceOperation {
            name: "new_reference",
            semantics: ReferenceOperationSemantics::new(
                vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                Vec::new(),
            ),
            forwarded: None,
        };
        let read = access("reference_read", ReferenceAccessMode::Read);
        let write = access("reference_swap", ReferenceAccessMode::Write);
        let freeze = access("freeze_reference", ReferenceAccessMode::Consume);

        // Allocation records no alias edge, because a fresh root is its own family. A view narrows the referent and
        // an identity edge does not, but narrowing is a property of the whole chain: an identity alias of a view
        // still names part of a root and therefore still cannot be consumed.
        let root = AtomId::new(1);
        let view = AtomId::new(2);
        let renamed_view = AtomId::new(3);
        let renamed_root = AtomId::new(4);
        let mut lifetimes = ReferenceLifetimes::default();
        assert_eq!(lifetimes.validate(&read, &[root]), Ok(()));
        lifetimes.record(&allocation, &[AtomId::new(0)], &[root]);
        assert_eq!(lifetimes.validate(&freeze, &[root]), Ok(()));
        lifetimes.record(&alias("reference_slice", ReferenceAliasKind::View), &[root], &[view]);
        lifetimes.record(&alias("rename", ReferenceAliasKind::Identity), &[view], &[renamed_view]);
        lifetimes.record(&alias("rename", ReferenceAliasKind::Identity), &[root], &[renamed_root]);
        assert_eq!(lifetimes.validate(&read, &[renamed_view]), Ok(()));
        let narrowed = "`freeze_reference` consumes a derived reference view, but consumption invalidates the whole \
                        alias family; consume the root handle instead";
        assert!(matches!(
            lifetimes.validate(&freeze, &[view]),
            Err(ProgramError::MalformedProgram(message)) if message == narrowed,
        ));
        assert!(matches!(
            lifetimes.validate(&freeze, &[renamed_view]),
            Err(ProgramError::MalformedProgram(message)) if message == narrowed,
        ));

        // Consuming a handle that does not narrow is accepted, including one reached through an identity edge and
        // one that no allocation in this region produced, which is where this rung is deliberately more permissive
        // than the standalone analysis.
        assert_eq!(lifetimes.validate(&freeze, &[renamed_root]), Ok(()));
        assert_eq!(lifetimes.validate(&freeze, &[AtomId::new(9)]), Ok(()));

        // Consumption invalidates every handle in the family, whichever handle performed it, and the diagnostic
        // names both the accessing operation and the one that consumed it.
        let consumed =
            |name, mode| format!("`{name}` {mode}s a reference whose alias family `freeze_reference` already consumed");
        lifetimes.record(&freeze, &[renamed_root], &[]);
        assert!(matches!(
            lifetimes.validate(&read, &[root]),
            Err(ProgramError::MalformedProgram(message)) if message == consumed("reference_read", "read"),
        ));
        assert!(matches!(
            lifetimes.validate(&write, &[view]),
            Err(ProgramError::MalformedProgram(message)) if message == consumed("reference_swap", "write"),
        ));

        // The narrowing rejection is checked before liveness, so a consumed view is reported as the misuse it is
        // rather than as an access to a dead family.
        assert!(matches!(
            lifetimes.validate(&freeze, &[renamed_view]),
            Err(ProgramError::MalformedProgram(message)) if message == narrowed,
        ));

        // An identity-forwarded output joins its operand's family even though the forwarding operation declares no
        // reference semantics at all, which is how a reference carried through a structured boundary stays tracked.
        let mut lifetimes = ReferenceLifetimes::default();
        let carry = AtomId::new(6);
        let carrier = TestReferenceOperation {
            name: "while",
            semantics: ReferenceOperationSemantics::default(),
            forwarded: Some((0, 0)),
        };
        lifetimes.record(&carrier, &[root], &[carry]);
        lifetimes.record(&freeze, &[root], &[]);
        assert!(matches!(
            lifetimes.validate(&read, &[carry]),
            Err(ProgramError::MalformedProgram(message)) if message == consumed("reference_read", "read"),
        ));

        // An unrelated root is untouched, and an out-of-range access index is left to the checks that own arity.
        assert_eq!(lifetimes.validate(&read, &[AtomId::new(7)]), Ok(()));
        assert_eq!(lifetimes.validate(&read, &[]), Ok(()));
    }
}
