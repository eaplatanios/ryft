use std::fmt::{Debug, Display};

use crate::programs::types::TypeError;

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
}
