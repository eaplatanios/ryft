use std::fmt::{Debug, Display};

use crate::programs::types::{Type, TypeError};

/// Nominal identity embedded in [`Program`](crate::Program) [`Type`](crate::Type) metadata. A nominal identity records
/// that repeated metadata occurrences denote the same unknown runtime quantity. It is distinct from a Single Static
/// Assignment (SSA) atom identity in that types retain it before any value-producing instruction exists, while
/// [`Region`](crate::Region) closure determines where the identity becomes available in the program graph.
pub trait TypeIdentity: Clone + Debug + Display + PartialEq + Eq {}

/// [`TypeIdentity`] used by [`Type`](crate::Type) families that carry no identity-bearing metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoIdentity {}

impl Display for NoIdentity {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = formatter;
        match *self {}
    }
}

impl TypeIdentity for NoIdentity {}

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
        if let Some((_, existing)) = self.replacements.iter().find(|(candidate, _)| candidate == &source) {
            if existing != &target {
                return Err(TypeError::invalid(format!(
                    "identity {source} is renamed to both {existing} and {target}",
                )));
            }
            return Ok(());
        }
        self.replacements.push((source, target));
        Ok(())
    }

    /// Returns the renamed identity, or a clone of `identity` when no replacement was registered.
    #[inline]
    pub fn rename(&self, identity: &I) -> I {
        self.replacements
            .iter()
            .find_map(|(source, target)| (source == identity).then(|| target.clone()))
            .unwrap_or_else(|| identity.clone())
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
    /// [`TypeIdentity`]s established by formal input types.
    input_identities: Vec<I>,

    /// [`TypeIdentity`]s established by [`Instruction`](crate::Instruction) results.
    internal_identities: Vec<I>,
}

impl<I: TypeIdentity> TypeIdentitySignature<I> {
    /// Creates a new [`TypeIdentitySignature`].
    #[inline]
    pub fn new(input_identities: Vec<I>, internal_identities: Vec<I>) -> Self {
        Self { input_identities, internal_identities }
    }

    /// Returns the [`TypeIdentity`]s established by formal input types.
    #[inline]
    pub fn input_identities(&self) -> &[I] {
        self.input_identities.as_slice()
    }

    /// Returns the [`TypeIdentity`]s established by [`Instruction`](crate::Instruction) results.
    #[inline]
    pub fn internal_identities(&self) -> &[I] {
        self.internal_identities.as_slice()
    }
}

/// Returns `true` if a cache entry imported for `cached_input_types` can safely serve `requested_input_types`. Exact
/// signatures can always share an entry. Non-exact signatures that contain any of the same live [`TypeIdentity`]s
/// cannot share because a permutation of those identities would alias a differently instantiated region. Signatures
/// with disjoint live identities can share when each signature can instantiate the other, proving that they differ
/// only by a safe alpha-renaming.
pub(crate) fn can_reuse_type_identity_instantiation<T: Type>(
    cached_input_types: &[T],
    requested_input_types: &[T],
) -> bool {
    if cached_input_types == requested_input_types {
        return true;
    }
    
    let mut identities_overlap = false;
    for cached_type in cached_input_types {
        cached_type.visit_identities(&mut |_, cached_identity| {
            for requested_type in requested_input_types {
                requested_type.visit_identities(&mut |_, requested_identity| {
                    identities_overlap |= cached_identity == requested_identity;
                });
            }
        });
    }
    
    if identities_overlap {
        return false;
    }
    
    T::derive_identity_renaming(cached_input_types, requested_input_types).is_ok()
        && T::derive_identity_renaming(requested_input_types, cached_input_types).is_ok()
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::types::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    /// Minimal nominal identity used to exercise simultaneous renaming.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestIdentity(&'static str);

    impl Display for TestIdentity {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl TypeIdentity for TestIdentity {}

    #[test]
    fn test_type_identity_renaming() {
        // Check that the type identity renaming is simultaneous.
        let first = TestIdentity("first");
        let second = TestIdentity("second");
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(first.clone(), second.clone()).unwrap();
        renaming.insert(second.clone(), first.clone()).unwrap();
        assert_eq!(renaming.rename(&first), second);
        assert_eq!(renaming.rename(&second), first);
        assert!(matches!(
            renaming.insert(TestIdentity("first"), TestIdentity("third")),
            Err(TypeError::Invalid { message })
                if message == "identity first is renamed to both second and third",
        ));
    }

    #[test]
    fn test_can_reuse_type_identity_instantiation() {
        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let cached_first = DimensionVariable::new("cached_first", bounds);
        let cached_second = DimensionVariable::new("cached_second", bounds);
        let requested_first = DimensionVariable::new("requested_first", bounds);
        let requested_second = DimensionVariable::new("requested_second", bounds);
        let array_type =
            |variable: DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)]));
        let cached_input_types = vec![array_type(cached_first.clone()), array_type(cached_second.clone())];

        // Exact signatures and structurally equivalent signatures with disjoint identities can share a cache entry.
        assert!(can_reuse_type_identity_instantiation(&cached_input_types, &cached_input_types));
        assert!(can_reuse_type_identity_instantiation(
            &cached_input_types,
            &[array_type(requested_first.clone()), array_type(requested_second)],
        ));

        // Overlapping permutations and structurally incompatible signatures must remain separate.
        assert!(!can_reuse_type_identity_instantiation(
            &cached_input_types,
            &[array_type(cached_second), array_type(cached_first)],
        ));
        assert!(!can_reuse_type_identity_instantiation(&cached_input_types, &[array_type(requested_first)],));
    }
}
