use std::borrow::Borrow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::programs::identities::{TypeIdentityPosition, TypeIdentityRenaming};
use crate::programs::types::{Type, TypeError, TypeRefinements};

/// [`Type`] that represents a reference to a [`Value`](crate::Value) whose [`Type`] is `T`. A reference type contains
/// only the type of the referenced value. Runtime resource identity belongs to [`Reference`](crate::Reference) and
/// therefore does not affect structural equality, hashing, or retained-program specialization. Reference compatibility
/// is exact in that a reference cannot implicitly broadcast or promote its storage, while refinement and identity
/// handling delegate to the referenced type. For [`ArrayType`](crate::ArrayType)s, exactness deliberately spans the
/// referenced value's optional layout, sharding, and memory metadata as well: the external-state mutation contract requires
/// exact physical referent compatibility, so a metadata-tolerant relation would overpromise.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceType<T: Type> {
    /// [`Type`] of the referenced value.
    referent: T,
}

impl<T: Type> ReferenceType<T> {
    /// Creates a new [`ReferenceType`].
    #[inline]
    pub fn new(referent: T) -> Self {
        Self { referent }
    }

    /// Returns the [`Type`] of the referenced [`Value`](crate::Value).
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

    #[inline]
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

/// Cross-occurrence [`TypeRefinements`] established for a complete [`ReferenceType`] signature.
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
        // Collecting the items is a shallow move as the referents themselves are delegated by borrow (i.e., `&T`
        // satisfies the `Borrow<T>` item bound), and so no referent is ever cloned on this type inference path.
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
        // Collecting the items is a shallow move as the referents themselves are delegated by borrow (i.e., `&T`
        // satisfies the `Borrow<T>` item bound), and so no referent is ever cloned on this type inference path.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let declared = declared.iter().map(|r#type| &r#type.borrow().referent);
        let actual = actual.iter().map(|r#type| &r#type.borrow().referent);
        self.referents.validate(declared, actual, closed_identities)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::parameters::Parameter;
    use crate::programs::identities::TypeIdentity;

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
        assert_eq!(Type::referent(&static_two), None);
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
}
