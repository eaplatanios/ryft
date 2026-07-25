use std::fmt::{Debug, Display};

use crate::programs::types::TypeError;

/// Nominal identity embedded in program type metadata.
///
/// A nominal identity records that repeated metadata occurrences denote the same unknown runtime quantity. It is
/// distinct from an SSA atom identity: types retain it before any value-producing instruction exists, while region
/// closure determines where the identity becomes available in the program graph.
pub trait TypeIdentity: Clone + Debug + Display + PartialEq + Eq {}

/// Identity used by type families that carry no identity-bearing metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoIdentity {}

impl Display for NoIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _ = formatter;
        match *self {}
    }
}

impl TypeIdentity for NoIdentity {}

/// Positional role of a nominal identity occurrence in program type metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IdentityPosition {
    /// The type occurrence owns the identity and can establish it at a program boundary or instruction result.
    Definition,

    /// The type occurrence refers to an identity established elsewhere in the program structure.
    Reference,
}

/// A simultaneous, capture-free renaming of live type identities.
///
/// Renamings are simultaneous: a target is never rewritten again merely because it also appears as a source. This
/// makes permutations and swaps well-defined without temporary identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeIdentityRenaming<I: TypeIdentity> {
    /// Source-to-target pairs in deterministic insertion order.
    replacements: Vec<(I, I)>,
}

impl<I: TypeIdentity> Default for TypeIdentityRenaming<I> {
    fn default() -> Self {
        Self { replacements: Vec::new() }
    }
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

    /// Returns whether applying this renaming leaves every identity unchanged.
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

/// Live identities available in one structurally closed region.
///
/// Boundary identities are ordered by first occurrence in formal input types. Internal identities follow in
/// instruction, result, and type-occurrence order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IdentitySignature<I: TypeIdentity> {
    /// Identities established by formal input types.
    input_identities: Vec<I>,

    /// Identities established by instruction results.
    internal_identities: Vec<I>,
}

impl<I: TypeIdentity> IdentitySignature<I> {
    /// Creates a closed signature from structurally validated source classes.
    pub(crate) fn from_parts(input_identities: Vec<I>, internal_identities: Vec<I>) -> Self {
        Self { input_identities, internal_identities }
    }

    /// Returns identities established by formal inputs.
    #[cfg(test)]
    #[inline]
    pub(crate) fn input_identities(&self) -> &[I] {
        self.input_identities.as_slice()
    }

    /// Returns identities established by instruction results.
    #[inline]
    pub(crate) fn internal_identities(&self) -> &[I] {
        self.internal_identities.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

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
    fn test_type_identity_renaming_is_simultaneous() {
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
}
