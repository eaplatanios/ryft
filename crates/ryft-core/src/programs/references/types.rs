use std::borrow::Borrow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::programs::identities::{NoIdentity, TypeIdentityPosition, TypeIdentityRenaming};
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

/// Referent family of [`Type`] universes that contain no references. It has no values, so a function
/// taking a borrowed [`NoReferent`] can never be called, and a [`ReferenceMemberType`] implementation whose
/// [`Referent`](ReferenceMemberType::Referent) is [`NoReferent`] satisfies reference-aware bounds without any runtime
/// rejection because the compiler proves every referent-consuming path unreachable. It follows [`NoIdentity`], which
/// plays the same role for identity-free type families.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum NoReferent {}

impl Display for NoReferent {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `NoReferent` has no variants, so this function can never be called. `Display` is implemented only because
        // `Type` requires it. Exhaustively matching the uninhabited `self` is the complete body (Rust accepts a match
        // with no arms as producing any type, including the `std::fmt::Result` return type), and nothing is ever
        // written. The formatter is unused as a consequence and it is referenced only to keep the conventional
        // `formatter` parameter name without an unused-variable warning.
        let _ = formatter;
        match *self {}
    }
}

impl Type for NoReferent {
    type Identity = NoIdentity;
    type Refinements = ();

    // `NoReferent` has no variants, so no value can ever reach these functions. Exhaustively matching that impossible
    // value lets Rust produce the required return types without fabricating semantics for a referent-free family.

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        let _ = other;
        match *self {}
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        let _ = other;
        match *self {}
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        match *self {}
    }

    #[inline]
    fn is_complex(&self) -> bool {
        match *self {}
    }
}

/// [`Type`] extension describing how a type universe represents references (i.e., the referent family that its
/// reference members wrap, the two projections between the universe and that family, and the conversions that embed
/// the family and its [`ReferenceType`] into the universe). It is implemented once per universe. A universe without
/// references implements it with the uninhabited [`NoReferent`], so that every function taking a borrowed referent
/// is unreachable for it and no runtime rejection is needed to express that the universe has no references.
///
/// # Canonical Referent
///
/// The associated [`Referent`](Self::Referent) makes "exactly one referent family per universe" a rule. The
/// per-operation conversion bounds of the reference operations never required this, so a universe could in principle
/// embed several [`ReferenceType`] families, but reverse-mode differentiation and reference discharge already treat a
/// universe as having one referent, and a second family would need a second set of cotangent constructors with no
/// consumer. A universe that embeds several [`ReferenceType`] families therefore cannot implement this trait.
///
/// # Laws
///
/// For every implementing universe with referent family `R`, the projections, [`Type::is_reference`], and the
/// conversions agree as follows, and each universe's tests check every law:
///
///   1. `is_reference()` holds exactly when `referent()` returns [`Some`].
///   2. `Self::from(ReferenceType::new(r)).referent() == Some(&r)` for every `r: R`.
///   3. `Self::from(r).as_referent() == Some(&r)` for every `r: R`, while `as_referent()` returns [`None`] for
///      reference members and for members that belong to neither the referent family nor its references (e.g.,
///      first-class dimensions in the array IR).
///   4. Wherever a universe also implements the borrowed `TryFrom<&Self>` conversions onto `&R` and
///      `&ReferenceType<R>`, those conversions succeed exactly where `as_referent()` and `referent()` return [`Some`],
///      respectively, and agree with them.
///   5. With `Referent = NoReferent`, `is_reference()` is `false` for every value and both projections return [`None`]
///      for every value.
///
/// # Projections As Functions
///
/// Both projections are functions rather than trait-level `where` clauses over the borrowed `TryFrom` conversions
/// because `where` clauses stated on a trait are not implied at its use sites, so every bound naming this trait would
/// have to restate them. The conversions remain per-operation bounds where operations need them, and the fourth law
/// ties them to the projections.
pub trait ReferenceMemberType: Type + From<Self::Referent> + From<ReferenceType<Self::Referent>> {
    /// Referent family of this universe's references, or [`NoReferent`] when the universe has none.
    type Referent: Type;

    /// Returns the referent of this type when it is a reference member of the universe, and [`None`] otherwise.
    fn referent(&self) -> Option<&Self::Referent>;

    /// Returns this type as a value of the referent family when it is the ordinary member of the universe that
    /// `From<Self::Referent>` produces, and [`None`] for reference members and for members of any other kind.
    fn as_referent(&self) -> Option<&Self::Referent>;
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::arrays::DataType;
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
    fn test_no_referent() {
        // `NoReferent` is uninhabited, so it, a reference type over it, and `Option`s of either are zero-sized: no
        // value of these types can exist, which is what lets referent-consuming code be proven unreachable.
        assert_eq!(size_of::<NoReferent>(), 0);
        assert_eq!(size_of::<Option<NoReferent>>(), 0);
        assert_eq!(size_of::<ReferenceType<NoReferent>>(), 0);
        assert_eq!(size_of::<Option<ReferenceType<NoReferent>>>(), 0);

        // `NoReferent` is a complete identity-free `Type`, so it can serve as the `Referent` of a universe and appear
        // wherever a `ReferenceMemberType` bound is used, both of which are checked here at compile time.
        fn assert_type<T: Type<Identity = NoIdentity, Refinements = ()>>() {}
        fn has_no_referent<T: ReferenceMemberType<Referent = NoReferent>>(r#type: &T) -> bool {
            r#type.referent().is_none() && r#type.as_referent().is_none() && !r#type.is_reference()
        }
        assert_type::<NoReferent>();
        assert!(has_no_referent(&DataType::F32));
    }
}
