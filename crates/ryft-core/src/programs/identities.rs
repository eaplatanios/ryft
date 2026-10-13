use std::fmt::{Debug, Display};

use crate::programs::types::TypeError;

/// Role of a [`TypeIdentity`] in one invocation's type-refinement facts. Declared and actual occurrences of the same
/// identity are tracked separately and related only when matching the invocation's types establishes a relationship.
/// The kind describes how an identity is used in a comparison; it is not a permanent property of the identity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeIdentityKind {
    /// Identity interpreted in the invoked program's declared scope, including its intermediate and output types.
    Declared,

    /// Identity interpreted in the scope of the actual supplied input types.
    Actual,
}

/// Nominal identity embedded in [`Program`](crate::Program) [`Type`](crate::Type) metadata. A nominal identity records
/// that repeated metadata occurrences denote the same unknown runtime quantity. It is distinct from a Single Static
/// Assignment (SSA) atom identity in that types retain it before any value-producing instruction exists, while
/// [`Region`](crate::Region) closure determines where the identity becomes available in the program graph.
pub trait TypeIdentity: Clone + Debug + Display + PartialEq + Eq {
    /// Returns a fresh [`TypeIdentity`] with the same user-facing metadata as `self`. [`Program`](crate::Program)
    /// relocation uses this function to rename [`TypeIdentity`]s defined inside an imported graph while preserving
    /// their metadata. The returned [`TypeIdentity`] must compare unequal to every existing identity, including `self`.
    fn fresh(&self) -> Self;
}

/// [`TypeIdentity`] used by [`Type`](crate::Type) families that carry no type-identity metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoIdentity {}

impl Display for NoIdentity {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = formatter;
        match *self {}
    }
}

impl TypeIdentity for NoIdentity {
    #[inline]
    fn fresh(&self) -> Self {
        // `NoIdentity` has no variants, so no value can ever reach this method. Exhaustively matching that impossible
        // value lets Rust produce the required return type without fabricating an identity for an identity-free type.
        match *self {}
    }
}

/// Positional role of a [`TypeIdentity`] occurrence in [`Program`](crate::Program) [`Type`](crate::Type) metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeIdentityPosition {
    /// The type occurrence owns the identity and can establish it at a program boundary or instruction result.
    Definition,

    /// The type occurrence refers to an identity established elsewhere in the program structure.
    Reference,
}

/// Represents a _simultaneous_, capture-free renaming of live [`TypeIdentity`]s. Renamings are simultaneous meaning
/// that a target is never rewritten again merely because it also appears as a source. This makes permutations and swaps
/// well-defined without using temporary identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeIdentityRenaming<I: TypeIdentity> {
    /// Source-to-target pairs in deterministic insertion order.
    replacements: Vec<(I, I)>,
}

impl<I: TypeIdentity> TypeIdentityRenaming<I> {
    /// Creates an empty identity renaming.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one source-to-target replacement, rejecting conflicting replacements for the same source.
    pub fn insert(&mut self, source: I, target: I) -> Result<(), TypeError> {
        if let Some(existing) = self.target(&source) {
            if existing != &target {
                return Err(TypeError::invalid(format!(
                    "type identity `{source}` is renamed to both `{existing}` and `{target}`",
                )));
            }
            return Ok(());
        }
        self.replacements.push((source, target));
        Ok(())
    }

    /// Adds a fresh replacement for `source`, rejecting a replacement that collides with any unavailable identity.
    /// Returns the validated replacement so that callers can make it unavailable to subsequent fresh replacements.
    pub fn insert_fresh(&mut self, source: I, unavailable: &[I]) -> Result<I, TypeError> {
        let target = source.fresh();
        if unavailable.contains(&target) {
            return Err(TypeError::invalid(format!(
                "fresh replacement for type identity `{source}` collides with an existing type identity",
            )));
        }
        self.insert(source, target.clone())?;
        Ok(target)
    }

    /// Returns the target registered for `source`, or [`None`] when no replacement was registered. Unlike
    /// [`Self::rename`], this function distinguishes an unregistered `source` from one that is renamed to itself.
    #[inline]
    pub fn target(&self, source: &I) -> Option<&I> {
        self.replacements.iter().find_map(|(candidate, target)| (candidate == source).then_some(target))
    }

    /// Returns the renamed identity, or a clone of `identity` when no replacement was registered.
    #[inline]
    pub fn rename(&self, identity: &I) -> I {
        self.target(identity).unwrap_or(identity).clone()
    }

    /// Returns `true` if applying this [`TypeIdentityRenaming`] leaves every [`TypeIdentity`] unchanged.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.replacements.iter().all(|(source, target)| source == target)
    }

    /// Returns the source-to-target replacements in insertion order.
    #[inline]
    pub fn replacements(&self) -> &[(I, I)] {
        self.replacements.as_slice()
    }
}

impl<I: TypeIdentity> Default for TypeIdentityRenaming<I> {
    #[inline]
    fn default() -> Self {
        Self { replacements: Vec::new() }
    }
}

/// Live [`TypeIdentity`]s available in one structurally closed [`Region`](crate::Region). Boundary identities are
/// ordered by first occurrence in formal input types. Internal identities follow in instruction, result, and
/// type-occurrence order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeIdentitySignature<I: TypeIdentity> {
    /// Complete closed [`TypeIdentity`] list which consists of the identities established by formal input types
    /// followed by identities established by [`Instruction`](crate::Instruction) results, stored as one ordered vector
    /// so boundary validation can consume the complete list without making any new allocations.
    identities: Vec<I>,

    /// Number of identities established by formal input types at the start of [`identities`](Self::identities).
    input_count: usize,
}

impl<I: TypeIdentity> TypeIdentitySignature<I> {
    /// Creates a new [`TypeIdentitySignature`] from one ordered identity list whose first `input_count` entries are
    /// established by formal input types and whose remaining entries are established internally.
    ///
    /// # Panics
    ///
    /// Panics if `input_count` exceeds the number of provided identities.
    ///
    /// # Parameters
    ///
    ///   - `identities`: Complete ordered list of input identities followed by internally established identities.
    ///   - `input_count`: Number of input identities at the start of `identities`.
    #[inline]
    pub fn new(identities: Vec<I>, input_count: usize) -> Self {
        assert!(
            input_count <= identities.len(),
            "input identity count {input_count} exceeds total identity count {}",
            identities.len(),
        );
        Self { identities, input_count }
    }

    /// Returns the [`TypeIdentity`]s established by formal input types.
    #[inline]
    pub fn input_identities(&self) -> &[I] {
        &self.identities[..self.input_count]
    }

    /// Returns the [`TypeIdentity`]s established by [`Instruction`](crate::Instruction) results.
    #[inline]
    pub fn internal_identities(&self) -> &[I] {
        &self.identities[self.input_count..]
    }

    /// Returns the complete closed [`TypeIdentity`] set which consists of the input signature identities
    /// followed by internally defined ones.
    #[inline]
    pub fn identities(&self) -> &[I] {
        self.identities.as_slice()
    }
}

/// A borrowed view of a connected group of scoped [`TypeIdentity`]s that must denote the same runtime quantity for
/// an invocation's type contract to hold. A scoped identity is an identity together with its [`TypeIdentityKind`].
/// Using the same identity object in both scopes does not, by itself, relate its declared and actual occurrences.
/// [`TypeRefinements::identity_classes`](crate::TypeRefinements::identity_classes) exposes these groups
/// so that consumers can preserve relationships between identities even when their values are unknown.
///
/// Matching an input's declared and actual types can relate their identities. Identities connected through one or more
/// such pairs belong to the same class. For example:
///
/// ```text
/// declared types: [dimension<n>, dimension<p>]
/// actual types:   [dimension<m>, dimension<m>]
///
/// first input:           declared n <-> actual m
/// second input:          declared p <-> actual m
///
/// single identity class: declared n <-> actual m <-> declared p
/// ```
///
/// Both connections reach the same actual `m`, so the class has declared members `[n, p]` and actual members `[m]`.
/// These member lists are not parallel signatures or positional pairs. In contrast, swapping two identities gives
/// two independent classes, because declared `m` and actual `m` are different scoped identities:
///
/// ```text
/// declared types:        [dimension<n>, dimension<m>]
/// actual types:          [dimension<m>, dimension<n>]
///
/// first input:           declared n <-> actual m
/// second input:          declared m <-> actual n
///
/// first identity class:  declared n <-> actual m
/// second identity class: declared m <-> actual n
/// ```
///
/// A class can relate identities without determining their value. Adding a third declared input `f32[n]` and actual
/// input `f32[2]` to the first example determines extent `2` for its entire class, making [`Self::is_exact`] `true`.
/// This means that the refinement facts determine one exact value. It does not mean the runtime check has executed,
/// that every containing type is fully static, or that consumers must erase symbolic identities.
///
/// Members are in first-observation order within each scope. Connections must be admitted by the type family;
/// unrelated identities do not join merely because they have equal exact values. All relationships and refinement
/// facts are conditional on the invocation satisfying its contract. This view neither performs nor proves that check.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TypeIdentityClass<'o, I: TypeIdentity> {
    /// Identities of the declared signature in this class.
    declared: &'o [I],

    /// Identities of the supplied signature in this class.
    actual: &'o [I],

    /// Whether the invocation fixes this class to an exact refinement.
    exact: bool,
}

impl<'o, I: TypeIdentity> TypeIdentityClass<'o, I> {
    /// Creates a [`TypeIdentityClass`] from its declared and actual members. `exact` indicates whether the refinement
    /// facts determine one exact value for those members.
    ///
    /// # Parameters
    ///
    ///   - `declared`: Declared identities, in first-observation order.
    ///   - `actual`: Supplied identities, in first-observation order.
    ///   - `exact`: Whether the invocation fixes this class to an exact refinement.
    #[inline]
    pub fn new(declared: &'o [I], actual: &'o [I], exact: bool) -> Self {
        Self { declared, actual, exact }
    }

    /// Returns the declared identities in first-observation order.
    #[inline]
    pub fn declared(&self) -> &'o [I] {
        self.declared
    }

    /// Returns the supplied identities in first-observation order.
    #[inline]
    pub fn actual(&self) -> &'o [I] {
        self.actual
    }

    /// Returns whether the invocation's refinement facts determine one exact value for this class's shared quantity.
    /// This fact is conditional on satisfying the invocation's contract, not evidence that its runtime check ran.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.exact
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::sync::Arc;

    use super::*;

    /// Minimal nominal identity used to exercise simultaneous renaming.
    #[derive(Clone, Debug)]
    struct TestIdentity {
        /// Diagnostic name.
        name: &'static str,

        /// Nominal identity token.
        token: Arc<()>,
    }

    impl TestIdentity {
        /// Creates a fresh test identity with `name`.
        fn new(name: &'static str) -> Self {
            Self { name, token: Arc::new(()) }
        }
    }

    impl Display for TestIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name)
        }
    }

    impl PartialEq for TestIdentity {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.token, &other.token)
        }
    }

    impl Eq for TestIdentity {}

    impl TypeIdentity for TestIdentity {
        fn fresh(&self) -> Self {
            Self::new(self.name)
        }
    }

    #[test]
    fn test_type_identity_renaming() {
        // Check that the type identity renaming is simultaneous.
        let first = TestIdentity::new("first");
        let second = TestIdentity::new("second");
        let fresh_first = first.fresh();
        assert_eq!(fresh_first.to_string(), first.to_string());
        assert_ne!(fresh_first, first);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(first.clone(), second.clone()).unwrap();
        renaming.insert(second.clone(), first.clone()).unwrap();
        assert_eq!(renaming.rename(&first), second);
        assert_eq!(renaming.rename(&second), first);

        // Check that targets distinguish unregistered identities from identities renamed to themselves.
        let third = TestIdentity::new("third");
        assert_eq!(renaming.target(&first), Some(&second));
        assert_eq!(renaming.target(&third), None);
        assert_eq!(renaming.rename(&third), third);
        let mut self_renaming = TypeIdentityRenaming::new();
        self_renaming.insert(third.clone(), third.clone()).unwrap();
        assert_eq!(self_renaming.target(&third), Some(&third));
        assert!(matches!(
            renaming.insert(first, TestIdentity::new("third")),
            Err(TypeError::Invalid { message })
                if message == "type identity `first` is renamed to both `second` and `third`",
        ));
    }

    /// Identity with a deliberately invalid freshness implementation.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CollidingIdentity {
        /// Diagnostic name.
        name: &'static str,
    }

    impl Display for CollidingIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name)
        }
    }

    impl TypeIdentity for CollidingIdentity {
        fn fresh(&self) -> Self {
            self.clone()
        }
    }

    #[test]
    fn test_type_identity_renaming_rejects_colliding_fresh_identity() {
        let identity = CollidingIdentity { name: "identity" };
        assert_eq!(
            TypeIdentityRenaming::new().insert_fresh(identity.clone(), std::slice::from_ref(&identity)),
            Err(TypeError::invalid(
                "fresh replacement for type identity `identity` collides with an existing type identity",
            )),
        );
    }

    #[test]
    fn test_type_identity_class() {
        let declared = [TestIdentity::new("declared")];
        let actual = [TestIdentity::new("actual")];
        let class = TypeIdentityClass::new(&declared, &actual, true);
        assert_eq!(class.declared(), &declared);
        assert_eq!(class.actual(), &actual);
        assert!(class.is_exact());
        assert!(!TypeIdentityClass::new(&declared, &[], false).is_exact());
    }
}
