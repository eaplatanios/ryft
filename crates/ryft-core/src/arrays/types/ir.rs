use std::borrow::Borrow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::sharding::ShardableType;
use crate::arrays::sharding::shardings::Sharding;
use crate::arrays::types::arrays::{ArrayType, ArrayTypeRefinements, AsArrayType};
use crate::arrays::types::dimensions::{
    AsDimensionType, Dimension, DimensionComparison, DimensionRefinementFact, DimensionRefinements, DimensionType,
    DimensionTypeRefinements, DimensionVariable,
};
use crate::parameters::Parameter;
use crate::programs::types::{derive_signature_identity_renaming, type_signature_pairs};
use crate::programs::{
    ReferenceMemberType, ReferenceType, Type, TypeError, TypeIdentityClass, TypeIdentityPosition, TypeIdentityRenaming,
    TypeRefinements, type_identities_needing_runtime_checks,
};

/// [`Type`] vocabulary of Ryft's array Intermediate Representation (IR), whose values may be ordinary arrays,
/// first-class runtime dimensions, or [`Reference`](crate::Reference)s to arrays. It is the type-level counterpart
/// of [`ArrayIrValue`](crate::ArrayIrValue), with one member per value kind.
///
/// The sum is a storage boundary rather than the contract ordinary primitives are written against. Array-only
/// [`Operation`](crate::Operation)s and transform rules keep consuming [`ArrayType`], dimension-only operations keep
/// consuming [`DimensionType`], and genuinely mixed operations consume this type directly. Those mixed operations
/// include shape-carrying operations with explicit dimension inputs and reference operations whose signatures cross
/// the array/reference boundary. [`From`] lifts each member type into the sum, and the borrowing [`TryFrom`]
/// implementations project it back out with a checked kind diagnostic. The same bridge backs the value-level
/// [`ValueProjection`](crate::ValueProjection) implementations.
///
/// All three members can carry [`DimensionVariable`] identities, so one renaming and refinement vocabulary spans
/// a complete signature. An [`ArrayType`] member and the referent of a [`ReferenceType`] member _reference_
/// the variables named by their dynamic axes, while a [`DimensionType`] member _defines_ its variable.
/// [`Type::derive_identity_renaming`] therefore checks a variable repeated across member kinds for consistency,
/// exactly as it does within one member kind.
///
/// Refer to the documentation of [`DimensionType`] for why runtime dimensions can be first-class typed values, the
/// three supported provenance categories, the single checked data-to-dimension gateway, and how those contracts enable
/// shape-polymorphic programs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayIrType {
    Array(ArrayType),
    Dimension(DimensionType),
    Reference(ReferenceType<ArrayType>),
}

impl ArrayIrType {
    /// Returns the [`Dimension`]s described by this [`ArrayIrType`] (i.e., the shape dimensions of an array or
    /// reference referent, or the single dimension represented by a first-class [`DimensionType`]). Stored dimension
    /// identities are preserved even when their bounds determine an exact extent.
    #[inline]
    pub(crate) fn dimensions(&self) -> &[Dimension] {
        match self {
            Self::Array(r#type) => r#type.shape().dimensions(),
            Self::Dimension(r#type) => std::slice::from_ref(r#type.dimension()),
            Self::Reference(r#type) => r#type.referent().shape().dimensions(),
        }
    }

    /// Projects a run of explicit extent input types into the [`Dimension`]s they define, in input order.
    ///
    /// Mixed shape-carrying operations derive their result shape from exactly this projection, so every mixed inference
    /// rule that consumes a trailing extent-input run shares one member-kind diagnostic for an input that is not a
    /// dimension.
    ///
    /// # Parameters
    ///
    ///   - `types`: Extent input types, in input order. The [`Borrow`] item bound lets type slices, owned [`Type`]
    ///     iterators, and borrowed [`Type`]s all be projected in place, so value-level callers need no intermediate
    ///     type collection.
    pub(crate) fn extents<T: IntoIterator<Item: Borrow<Self>>>(types: T) -> Result<Vec<Dimension>, TypeError> {
        types
            .into_iter()
            .map(|r#type| <&DimensionType>::try_from(r#type.borrow()).map(DimensionType::simplified_dimension))
            .collect()
    }

    /// Returns this array IR member's diagnostic kind name.
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Array(_) => "array",
            Self::Dimension(_) => "dimension",
            Self::Reference(_) => "reference",
        }
    }
}

impl Display for ArrayIrType {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(r#type) => Display::fmt(r#type, formatter),
            Self::Dimension(r#type) => Display::fmt(r#type, formatter),
            Self::Reference(r#type) => Display::fmt(r#type, formatter),
        }
    }
}

impl From<ArrayType> for ArrayIrType {
    #[inline]
    fn from(r#type: ArrayType) -> Self {
        Self::Array(r#type)
    }
}

impl From<DimensionType> for ArrayIrType {
    #[inline]
    fn from(r#type: DimensionType) -> Self {
        Self::Dimension(r#type)
    }
}

impl From<ReferenceType<ArrayType>> for ArrayIrType {
    #[inline]
    fn from(r#type: ReferenceType<ArrayType>) -> Self {
        Self::Reference(r#type)
    }
}

impl<'t> TryFrom<&'t ArrayIrType> for &'t ArrayType {
    type Error = TypeError;

    #[inline]
    fn try_from(r#type: &'t ArrayIrType) -> Result<Self, Self::Error> {
        match r#type {
            ArrayIrType::Array(r#type) => Ok(r#type),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind_name()))),
        }
    }
}

impl<'t> TryFrom<&'t ArrayIrType> for &'t DimensionType {
    type Error = TypeError;

    #[inline]
    fn try_from(r#type: &'t ArrayIrType) -> Result<Self, Self::Error> {
        match r#type {
            ArrayIrType::Dimension(r#type) => Ok(r#type),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind_name()))),
        }
    }
}

impl<'t> TryFrom<&'t ArrayIrType> for &'t ReferenceType<ArrayType> {
    type Error = TypeError;

    #[inline]
    fn try_from(r#type: &'t ArrayIrType) -> Result<Self, Self::Error> {
        match r#type {
            ArrayIrType::Reference(r#type) => Ok(r#type),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind_name()))),
        }
    }
}

impl Type for ArrayIrType {
    type Identity = DimensionVariable;
    type Refinements = ArrayIrTypeRefinements;

    fn validate_zero(&self) -> Result<(), TypeError> {
        match self {
            Self::Array(r#type) => r#type.validate_zero(),
            Self::Dimension(_) => Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type")),
            Self::Reference(r#type) => Err(TypeError::invalid(format!(
                "cannot materialize a zero for reference type `{}`; a reference denotes an allocation and has \
                 no zero value, so tangent and cotangent references are allocated by the differentiation rules",
                r#type,
            ))),
        }
    }

    fn validate_one(&self) -> Result<(), TypeError> {
        match self {
            Self::Array(r#type) => r#type.validate_one(),
            Self::Dimension(_) => Err(TypeError::invalid("cannot materialize a one for a first-class dimension type")),
            Self::Reference(r#type) => Err(TypeError::invalid(format!(
                "cannot materialize a one for reference type `{}`; a reference denotes an allocation and has \
                 no one value, so tangent and cotangent references are allocated by the differentiation rules",
                r#type,
            ))),
        }
    }

    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        let (storage, definition) = match self {
            Self::Array(r#type) => (r#type.shape().dimensions(), None),
            Self::Dimension(r#type) => (&[][..], r#type.variable()),
            Self::Reference(r#type) => (r#type.referent().shape().dimensions(), None),
        };
        let mut storage = storage.iter();
        let mut definition = definition;
        std::iter::from_fn(move || {
            if let Some(identity) = storage.find_map(Dimension::variable) {
                Some((TypeIdentityPosition::Reference, identity))
            } else {
                definition.take().map(|identity| (TypeIdentityPosition::Definition, identity))
            }
        })
    }

    fn derive_identity_renaming<
        'o,
        D: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
    >(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 'o,
    {
        derive_signature_identity_renaming(
            declared,
            actual,
            |declared, actual| {
                ArrayIrTypeRefinements::refinement_facts(declared, actual, DimensionComparison::IdentityRenaming)
            },
            DimensionRefinements::derive_identity_renaming,
        )
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        match self {
            Self::Array(r#type) => Ok(Self::Array(r#type.rename_identities(renaming)?)),
            Self::Dimension(r#type) => Ok(Self::Dimension(r#type.rename_identities(renaming)?)),
            Self::Reference(r#type) => Ok(Self::Reference(r#type.rename_identities(renaming)?)),
        }
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Array(left), Self::Array(right)) => left.is_compatible_with(right),
            (Self::Dimension(left), Self::Dimension(right)) => left.is_compatible_with(right),
            (Self::Reference(left), Self::Reference(right)) => left.is_compatible_with(right),
            _ => false,
        }
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Array(left), Self::Array(right)) => left.is_refined_by(right),
            (Self::Dimension(left), Self::Dimension(right)) => left.is_refined_by(right),
            (Self::Reference(left), Self::Reference(right)) => left.is_refined_by(right),
            _ => false,
        }
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        match self {
            Self::Array(r#type) => r#type.is_scalar(),
            Self::Dimension(r#type) => r#type.is_scalar(),
            Self::Reference(_) => false,
        }
    }

    #[inline]
    fn is_complex(&self) -> bool {
        match self {
            Self::Array(r#type) => r#type.is_complex(),
            Self::Dimension(r#type) => r#type.is_complex(),
            Self::Reference(_) => false,
        }
    }

    #[inline]
    fn is_reference(&self) -> bool {
        matches!(self, Self::Reference(_))
    }
}

impl ShardableType for ArrayIrType {
    #[inline]
    fn sharding(&self) -> Option<&Sharding> {
        match self {
            Self::Array(r#type) => r#type.sharding(),
            Self::Dimension(r#type) => r#type.sharding(),
            Self::Reference(r#type) => r#type.sharding(),
        }
    }
}

impl AsArrayType for ArrayIrType {
    #[inline]
    fn as_array_type(&self) -> Result<&ArrayType, TypeError> {
        self.try_into()
    }
}

impl AsDimensionType for ArrayIrType {
    #[inline]
    fn as_dimension_type(&self) -> Result<&DimensionType, TypeError> {
        self.try_into()
    }
}

impl ReferenceMemberType for ArrayIrType {
    type Referent = ArrayType;

    #[inline]
    fn referent(&self) -> Option<&ArrayType> {
        match self {
            Self::Reference(r#type) => Some(r#type.referent()),
            Self::Array(_) | Self::Dimension(_) => None,
        }
    }

    #[inline]
    fn as_referent(&self) -> Option<&ArrayType> {
        match self {
            Self::Array(r#type) => Some(r#type),
            Self::Dimension(_) | Self::Reference(_) => None,
        }
    }
}

/// [`TypeRefinements`] for one complete [`ArrayIrType`] signature. Array members and reference referents follow the
/// [`ArrayTypeRefinements`] rules and share the [`DimensionRefinements`] declared and actual scopes with first-class
/// dimension members, so array shapes and first-class dimension inputs must agree. A dimension member must have actual
/// bounds within its declared bounds. Matching local declarations relates distinct dimension and array-axis identities,
/// sharing exact facts across those links; ordinary nominal input admission still restricts array axes. A dimension
/// member with singleton bounds contributes its exact extent. Wider dimension bounds supply no concrete fact; their
/// declared variable belongs to the boundary's closed identity set (refer to the documentation of
/// [`TypeIdentitySignature`](crate::TypeIdentitySignature) for more information), which lets
/// output validation record its first concrete extent and reject inconsistent later extents.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayIrTypeRefinements {
    /// [`ArrayTypeRefinements`] shared by array members and reference referents across the complete signature.
    arrays: ArrayTypeRefinements,
}

impl ArrayIrTypeRefinements {
    /// Returns the [`DimensionRefinementFact`]s for one declared and one actual [`ArrayIrType`], or their comparison
    /// error. Validates matching kinds before comparing arrays, reference referents, or dimension values. Arrays and
    /// reference referents use [`ArrayTypeRefinements::refinement_facts`] with the requested [`DimensionComparison`].
    /// Dimension values use [`DimensionTypeRefinements::refinement_facts`] with bounds-based
    /// [`DimensionComparison::LocalDeclarations`] rules, or [`DimensionComparison::IdentityRenaming`] when requested.
    /// Structural comparisons preserve substitutions rather than infer conditional exact extents from supplied
    /// variables.
    fn refinement_facts<'o>(
        declared: &'o ArrayIrType,
        actual: &'o ArrayIrType,
        comparison: DimensionComparison,
    ) -> impl Iterator<Item = Result<DimensionRefinementFact<'o>, TypeError>> {
        let (array_pair, dimension_pair, error) = match (declared, actual) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) => (Some((declared, actual)), None, None),
            (ArrayIrType::Dimension(declared), ArrayIrType::Dimension(actual)) => {
                (None, Some((declared, actual)), None)
            }
            (ArrayIrType::Reference(declared), ArrayIrType::Reference(actual)) => {
                (Some((declared.referent(), actual.referent())), None, None)
            }
            // Keep cross-kind arms keyed on the declared variant so new members require an explicit policy.
            (ArrayIrType::Array(_), actual) => (
                None,
                None,
                Some(TypeError::invalid(format!("expected array type but got {} type", actual.kind_name()))),
            ),
            (ArrayIrType::Dimension(_), actual) => (
                None,
                None,
                Some(TypeError::invalid(format!("expected dimension type but got {} type", actual.kind_name()))),
            ),
            (ArrayIrType::Reference(_), actual) => (
                None,
                None,
                Some(TypeError::invalid(format!("expected reference type but got {} type", actual.kind_name()))),
            ),
        };
        error
            .into_iter()
            .map(Err)
            .chain(array_pair.into_iter().flat_map(move |(declared, actual)| {
                ArrayTypeRefinements::refinement_facts(declared, actual, comparison)
            }))
            .chain(dimension_pair.into_iter().flat_map(move |(declared, actual)| {
                let comparison = match comparison {
                    DimensionComparison::Refinement | DimensionComparison::LocalDeclarations => {
                        DimensionComparison::LocalDeclarations
                    }
                    DimensionComparison::IdentityRenaming => DimensionComparison::IdentityRenaming,
                };
                DimensionTypeRefinements::refinement_facts(declared, actual, comparison)
            }))
    }
}

impl TypeRefinements<ArrayIrType> for ArrayIrTypeRefinements {
    fn new<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayIrType>,
        A::Item: Borrow<ArrayIrType>,
    {
        // Member admission follows its existing array, dimension, or reference rules. Comparing the complete signature
        // then checks variables shared within and between all member kinds.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        type_signature_pairs(declared.iter(), actual.iter())?.try_for_each(|(declared, actual)| {
            Self::refinement_facts(declared.borrow(), actual.borrow(), DimensionComparison::Refinement)
                .try_for_each(|refinement_fact| refinement_fact.map(|_| ()))
        })?;
        Self::from_local_declarations(declared, actual)
    }

    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayIrType>,
        A::Item: Borrow<ArrayIrType>,
    {
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let refinements = DimensionRefinements::from_refinement_facts(
            type_signature_pairs(declared.iter().map(Borrow::borrow), actual.iter().map(Borrow::borrow))?.flat_map(
                |(declared, actual)| Self::refinement_facts(declared, actual, DimensionComparison::LocalDeclarations),
            ),
        )?;
        let actual = actual.iter().map(|r#type| Some(r#type.borrow())).collect::<Vec<_>>();
        let unproven = type_identities_needing_runtime_checks::<ArrayIrType, _>(&declared, &actual)?;
        Ok(Self { arrays: ArrayTypeRefinements::from(refinements.with_identities_needing_runtime_checks(unproven)) })
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, <ArrayIrType as Type>::Identity>>
    where
        <ArrayIrType as Type>::Identity: 'o,
    {
        self.arrays.identity_classes()
    }

    #[inline]
    fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        self.arrays.identities_needing_runtime_checks()
    }

    fn validate_refinement<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayIrType>,
        A::Item: Borrow<ArrayIrType>,
    {
        // Output validation uses declared-scope facts and never records actual-scope facts or correspondences.
        let mut refinements = self.arrays.dimension_refinements().clone();
        type_signature_pairs(declared, actual)?.try_for_each(|(declared, actual)| {
            Self::refinement_facts(declared.borrow(), actual.borrow(), DimensionComparison::Refinement).try_for_each(
                |refinement_fact| match refinement_fact? {
                    DimensionRefinementFact::Declared(variable, extent) => {
                        refinements.validate_declared_extent(variable, extent, closed_identities)
                    }
                    DimensionRefinementFact::Actual(..) | DimensionRefinementFact::Correspondence(..) => Ok(()),
                },
            )
        })
    }

    fn refine_declared(
        &self,
        r#type: &ArrayIrType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<ArrayIrType, TypeError> {
        // A first-class dimension type whose variable is bound to a static extent describes a static dimension,
        // like an array axis of that variable.
        Ok(match r#type {
            ArrayIrType::Array(r#type) => {
                ArrayIrType::Array(self.arrays.refine_declared(r#type, preserved_identities)?)
            }
            ArrayIrType::Dimension(r#type) => ArrayIrType::Dimension(DimensionType::from(
                self.arrays
                    .dimension_refinements()
                    .refine_declared_dimension(r#type.dimension(), preserved_identities),
            )),
            ArrayIrType::Reference(r#type) => ArrayIrType::Reference(ReferenceType::new(
                self.arrays.refine_declared(r#type.referent(), preserved_identities)?,
            )),
        })
    }

    fn refine_actual(
        &self,
        r#type: &ArrayIrType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<ArrayIrType, TypeError> {
        Ok(match r#type {
            ArrayIrType::Array(r#type) => ArrayIrType::Array(self.arrays.refine_actual(r#type, preserved_identities)?),
            ArrayIrType::Dimension(r#type) => ArrayIrType::Dimension(DimensionType::from(
                self.arrays
                    .dimension_refinements()
                    .refine_actual_dimension(r#type.dimension(), preserved_identities),
            )),
            ArrayIrType::Reference(r#type) => ArrayIrType::Reference(ReferenceType::new(
                self.arrays.refine_actual(r#type.referent(), preserved_identities)?,
            )),
        })
    }

    fn retype_actual(
        &self,
        declared: &ArrayIrType,
        actual: &ArrayIrType,
        preferred_identities: &[DimensionVariable],
    ) -> Result<ArrayIrType, TypeError> {
        Ok(match (declared, actual) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) => {
                ArrayIrType::Array(self.arrays.retype_actual(declared, actual, preferred_identities)?)
            }
            (ArrayIrType::Dimension(declared), ArrayIrType::Dimension(actual)) => ArrayIrType::Dimension(
                DimensionType::from(self.arrays.dimension_refinements().retype_actual_dimension(
                    declared.dimension(),
                    actual.dimension(),
                    preferred_identities,
                )),
            ),
            (ArrayIrType::Reference(declared), ArrayIrType::Reference(actual)) => {
                ArrayIrType::Reference(ReferenceType::new(self.arrays.retype_actual(
                    declared.referent(),
                    actual.referent(),
                    preferred_identities,
                )?))
            }
            _ => return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::types::data::DataType::F32;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionError, Shape};

    use super::*;

    #[test]
    fn test_array_ir_type() {
        let declared_variable = DimensionVariable::new("declared", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual_variable = DimensionVariable::new("actual", DimensionBounds::non_negative(Some(4)).unwrap());

        // Extent projection accepts an empty run, preserves input order, recognizes singleton-bounded dimensions as
        // static, retains wider-bounded dimensions as dynamic, and rejects non-dimension members.
        let exact_variable = DimensionVariable::new("exact", DimensionBounds::new(4, Some(5)).unwrap());
        let extent_types = [
            ArrayIrType::Dimension(DimensionType::from(exact_variable)),
            ArrayIrType::Dimension(DimensionType::from(declared_variable.clone())),
        ];
        assert_eq!(ArrayIrType::extents(std::iter::empty::<ArrayIrType>()), Ok(Vec::new()));
        assert_eq!(
            ArrayIrType::extents(&extent_types),
            Ok(vec![Dimension::Static(4), Dimension::Dynamic(declared_variable.clone())]),
        );
        assert_eq!(
            ArrayIrType::extents([ArrayIrType::Array(ArrayType::scalar(F32))]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        // One identity may be defined by a dimension member and referenced by an array member. Signature matching
        // must preserve that relationship under one consistent renaming and reject arity or member-kind mismatches.
        let declared = [
            ArrayIrType::Dimension(DimensionType::from(declared_variable.clone())),
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(declared_variable.clone())]))),
        ];
        let actual = [
            ArrayIrType::Dimension(DimensionType::from(actual_variable.clone())),
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(actual_variable.clone())]))),
        ];
        assert_eq!(
            declared
                .iter()
                .flat_map(ArrayIrType::identities)
                .map(|(position, identity)| (position, identity.clone()))
                .collect::<Vec<_>>(),
            vec![
                (TypeIdentityPosition::Definition, declared_variable.clone()),
                (TypeIdentityPosition::Reference, declared_variable.clone()),
            ],
        );

        let renaming = ArrayIrType::derive_identity_renaming(&declared, &actual).unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
        let declared_with_reference = [
            declared[1].clone(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
                F32,
                Shape::new(vec![Dimension::Dynamic(declared_variable.clone())]),
            ))),
        ];
        let actual_with_reference = [
            actual[1].clone(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
                F32,
                Shape::new(vec![Dimension::Dynamic(actual_variable.clone())]),
            ))),
        ];
        let renaming = ArrayIrType::derive_identity_renaming(&declared_with_reference, &actual_with_reference).unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
        let declared_complete = [declared[0].clone(), declared[1].clone(), declared_with_reference[1].clone()];
        let actual_complete = [actual[0].clone(), actual[1].clone(), actual_with_reference[1].clone()];
        let renaming = ArrayIrType::derive_identity_renaming(&declared_complete, &actual_complete).unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
        assert_eq!(
            ArrayIrType::derive_identity_renaming(&declared, &actual[..1]),
            Err(TypeError::invalid("declared type count 2 does not match actual type count 1")),
        );
        assert_eq!(
            ArrayIrType::derive_identity_renaming(&declared[..1], &[ArrayIrType::Array(ArrayType::scalar(F32))]),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        // Repeated occurrences of one dynamic array variable record one exact extent and reject a later
        // conflicting extent.
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared_array =
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])));
        let actual_two = ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])));
        let actual_three = ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)])));
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            [declared_array.clone(), declared_array.clone()],
            [actual_two.clone(), actual_two.clone()],
        )
        .unwrap();
        assert_eq!(refinements.validate_refinement([declared_array.clone()], [actual_two.clone()], &[]), Ok(()));
        let error = ArrayIrTypeRefinements::from_local_declarations(
            [declared_array.clone(), declared_array.clone()],
            [ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]))), actual_three],
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        // Array members and reference referents share one identity-renaming and refinement accumulator. Repeating an
        // identity across those two member kinds must therefore preserve one renaming and reject conflicting concrete
        // extents exactly as repeated ordinary arrays do.
        let declared_reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Dynamic(batch.clone())]),
        )));
        let actual_reference_two =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]))));
        let actual_reference_three =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]))));
        let mixed_refinements = ArrayIrTypeRefinements::from_local_declarations(
            [declared_array.clone(), declared_reference.clone()],
            [actual_two.clone(), actual_reference_two],
        )
        .unwrap();
        assert_eq!(
            mixed_refinements.validate_refinement(
                [declared_array.clone(), declared_reference.clone()],
                [
                    actual_two.clone(),
                    ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
                        F32,
                        Shape::new(vec![Dimension::Static(2)]),
                    )))
                ],
                &[],
            ),
            Ok(()),
        );
        let error = ArrayIrTypeRefinements::from_local_declarations(
            [declared_array.clone(), declared_reference],
            [actual_two.clone(), actual_reference_three],
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        // A dimension member with wider bounds contributes no concrete fact. Its variable instead belongs to the
        // boundary's closed identity signature, so output validation may record its concrete extent on the first
        // refinement fact and must reject an inconsistent repeated refinement fact within the same validated signature.
        let declared_dimension = ArrayIrType::Dimension(DimensionType::from(batch.clone()));
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            std::slice::from_ref(&declared_dimension),
            std::slice::from_ref(&declared_dimension),
        )
        .unwrap();
        let closed_identities = [batch.clone()];
        assert_eq!(
            refinements.validate_refinement(
                [declared_array.clone(), declared_array.clone()],
                [
                    ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]))),
                    ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]))),
                ],
                &closed_identities,
            ),
            Ok(()),
        );
        let error = refinements
            .validate_refinement(
                [declared_array.clone(), declared_array.clone()],
                [
                    ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]))),
                    ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]))),
                ],
                &closed_identities,
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        // Singleton dimension bounds and static array axes constrain the same boundary variable. Record their
        // agreement in either signature order and reject contradictory extents before executing any instruction.
        for extent in [2, 3] {
            let exact_dimension = ArrayIrType::Dimension(DimensionType::new(
                "exact",
                DimensionBounds::new(extent, Some(extent + 1)).unwrap(),
            ));
            for (declared, actual) in [
                ([declared_array.clone(), declared_dimension.clone()], [actual_two.clone(), exact_dimension.clone()]),
                ([declared_dimension.clone(), declared_array.clone()], [exact_dimension.clone(), actual_two.clone()]),
            ] {
                let result = ArrayIrTypeRefinements::from_local_declarations(declared, actual);
                if extent == 2 {
                    assert_eq!(
                        result.unwrap().validate_refinement([declared_array.clone()], [actual_two.clone()], &[]),
                        Ok(())
                    );
                } else {
                    assert!(matches!(
                        result.unwrap_err().downcast_custom::<DimensionError>(),
                        Some(DimensionError::InputDimensionMismatch { dimension, expected, actual })
                            if dimension == "batch" && ((*expected, *actual) == (2, 3) || (*expected, *actual) == (3, 2)),
                    ));
                }
            }
            let exact_refinements =
                ArrayIrTypeRefinements::from_local_declarations([declared_dimension.clone()], [exact_dimension])
                    .unwrap();
            let result = exact_refinements.validate_refinement([declared_array.clone()], [actual_two.clone()], &[]);
            if extent == 2 {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(
                    result.unwrap_err().downcast_custom::<DimensionError>(),
                    Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 3,
                        actual: 2,
                    }),
                );
            }
        }

        // An identity outside the boundary's closed signature stays rejected, and an internally defined identity may
        // receive its first fact exactly like an input-signature identity.
        let unrelated = DimensionVariable::new("unrelated", DimensionBounds::non_negative(Some(8)).unwrap());
        let unrelated_array =
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(unrelated.clone())])));
        assert_eq!(
            refinements.validate_refinement(
                std::slice::from_ref(&unrelated_array),
                [ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])))],
                &closed_identities,
            ),
            Err(TypeError::invalid("dimension identity unrelated does not belong to the validated boundary signature")),
        );
        assert_eq!(
            refinements.validate_refinement(
                std::slice::from_ref(&unrelated_array),
                [ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])))],
                std::slice::from_ref(&unrelated),
            ),
            Ok(()),
        );
    }

    #[test]
    fn test_array_ir_type_dimensions() {
        let dimension = DimensionType::new("n", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimensions = vec![Dimension::Static(2), dimension.simplified_dimension()];
        let array = ArrayType::new(F32, Shape::new(dimensions.clone()));
        assert_eq!(ArrayIrType::Array(array.clone()).dimensions(), dimensions.as_slice());
        assert_eq!(ArrayIrType::Dimension(dimension.clone()).dimensions(), std::slice::from_ref(dimension.dimension()));
        assert_eq!(ArrayIrType::Reference(ReferenceType::new(array)).dimensions(), dimensions.as_slice());
        assert_eq!(ArrayIrType::Array(ArrayType::scalar(F32)).dimensions(), &[]);
        let fixed = DimensionType::new("fixed", DimensionBounds::new(3, Some(4)).unwrap());
        assert_eq!(ArrayIrType::Dimension(fixed.clone()).dimensions(), &[Dimension::Dynamic(fixed.to_variable())]);
        assert_eq!(
            ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3))).dimensions(),
            &[Dimension::Static(3)],
        );
    }

    #[test]
    fn test_array_ir_type_derive_identity_renaming_rejects_retained_and_statically_bound_definition() {
        let variable = DimensionVariable::new("n", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension = ArrayIrType::Dimension(DimensionType::from(variable.clone()));
        let dynamic_type = ArrayType::new(F32, Shape::new(vec![variable.clone().into()]));
        let static_type = ArrayType::new_static(F32, [3]);

        let error = TypeError::invalid(
            "dimension variable n is renamed to n by one signature member and bound to static extent 3 by another",
        );
        // Refinement accepts a concrete array or reference alongside a wide nominal dimension definition.
        // Pure substitution remains strict because it cannot retain that cross-input runtime precondition.
        for (declared_member, actual_member) in [
            (ArrayIrType::Array(dynamic_type.clone()), ArrayIrType::Array(static_type.clone())),
            (
                ArrayIrType::Reference(ReferenceType::new(dynamic_type)),
                ArrayIrType::Reference(ReferenceType::new(static_type)),
            ),
        ] {
            for (declared, actual) in [
                ([dimension.clone(), declared_member.clone()], [dimension.clone(), actual_member.clone()]),
                ([declared_member, dimension.clone()], [actual_member, dimension.clone()]),
            ] {
                assert_eq!(ArrayIrType::derive_identity_renaming(&declared, &actual), Err(error.clone()));
                ArrayIrTypeRefinements::from_local_declarations(&declared, &actual).unwrap();
            }
        }
    }

    #[test]
    fn test_array_ir_type_validate_zero() {
        let array = ArrayType::scalar(F32);
        assert_eq!(ArrayIrType::Array(array.clone()).validate_zero(), Ok(()));
        let dimension = DimensionType::new("size", DimensionBounds::unbounded());
        assert_eq!(
            ArrayIrType::Dimension(dimension).validate_zero(),
            Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type")),
        );
        let reference = ReferenceType::new(array);
        assert_eq!(
            ArrayIrType::Reference(reference.clone()).validate_zero(),
            Err(TypeError::invalid(format!(
                "cannot materialize a zero for reference type `{reference}`; a reference denotes an allocation and has \
                 no zero value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_array_ir_type_validate_one() {
        let array = ArrayType::scalar(F32);
        assert_eq!(ArrayIrType::Array(array.clone()).validate_one(), Ok(()));
        let dimension = DimensionType::new("size", DimensionBounds::unbounded());
        assert_eq!(
            ArrayIrType::Dimension(dimension).validate_one(),
            Err(TypeError::invalid("cannot materialize a one for a first-class dimension type")),
        );
        let reference = ReferenceType::new(array);
        assert_eq!(
            ArrayIrType::Reference(reference.clone()).validate_one(),
            Err(TypeError::invalid(format!(
                "cannot materialize a one for reference type `{reference}`; a reference denotes an allocation and has \
                 no one value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_array_ir_type_projection() {
        // An array member projects back to its exact borrowed array type and rejects projection as a dimension.
        let array = ArrayType::new(F32, Shape::scalar());
        let stored = ArrayIrType::from(array.clone());
        assert_eq!(<&ArrayType>::try_from(&stored), Ok(&array));
        assert!(!stored.is_reference());
        assert_eq!(stored.referent(), None);
        assert_eq!(
            <&DimensionType>::try_from(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        assert_eq!(
            <&ReferenceType<ArrayType>>::try_from(&stored),
            Err(TypeError::invalid("expected reference type but got array type")),
        );

        // A dimension member provides the symmetric successful dimension projection and array-kind diagnostic.
        let dimension = DimensionType::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let stored = ArrayIrType::from(dimension.clone());
        assert_eq!(<&DimensionType>::try_from(&stored), Ok(&dimension));
        assert!(!stored.is_reference());
        assert_eq!(stored.referent(), None);
        assert_eq!(
            <&ArrayType>::try_from(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            <&ReferenceType<ArrayType>>::try_from(&stored),
            Err(TypeError::invalid("expected reference type but got dimension type")),
        );

        // A reference member projects only as its exact reference type and never as the referent array type.
        let reference = ReferenceType::new(array.clone());
        let stored = ArrayIrType::from(reference.clone());
        assert_eq!(<&ReferenceType<ArrayType>>::try_from(&stored), Ok(&reference));
        assert!(!stored.is_scalar());
        assert!(!stored.is_complex());
        assert!(stored.is_reference());
        assert_eq!(stored.referent(), Some(&array));
        assert!(!stored.is_compatible_with(&ArrayIrType::Array(array.clone())));
        assert!(!ArrayIrType::Array(array.clone()).is_compatible_with(&stored));
        assert!(!stored.is_refined_by(&ArrayIrType::Array(array.clone())));
        assert!(!ArrayIrType::Array(array).is_refined_by(&stored));
        assert_eq!(
            <&ArrayType>::try_from(&stored),
            Err(TypeError::invalid("expected array type but got reference type")),
        );
        assert_eq!(
            <&DimensionType>::try_from(&stored),
            Err(TypeError::invalid("expected dimension type but got reference type")),
        );
    }

    #[test]
    fn test_array_ir_type_referent() {
        let array = ArrayType::new_static(F32, [3]);
        let reference_member = ArrayIrType::from(ReferenceType::new(array.clone()));
        let array_member = ArrayIrType::from(array.clone());
        let dimension_member = ArrayIrType::from(DimensionType::new("extent", DimensionBounds::unbounded()));
        assert_eq!(reference_member.referent(), Some(&array));
        assert_eq!(array_member.referent(), None);
        assert_eq!(dimension_member.referent(), None);
        assert!(reference_member.is_reference());
        assert!(!array_member.is_reference());
        assert!(!dimension_member.is_reference());
        assert_eq!(
            reference_member.referent(),
            <&ReferenceType<ArrayType>>::try_from(&reference_member).ok().map(|reference| reference.referent()),
        );
    }

    #[test]
    fn test_array_ir_type_as_referent() {
        let array = ArrayType::new_static(F32, [3]);
        let array_member = ArrayIrType::from(array.clone());
        let reference_member = ArrayIrType::from(ReferenceType::new(array.clone()));
        let dimension_member = ArrayIrType::from(DimensionType::new("extent", DimensionBounds::unbounded()));
        assert_eq!(array_member.as_referent(), Some(&array));
        assert_eq!(reference_member.as_referent(), None);
        assert_eq!(dimension_member.as_referent(), None);
        assert_eq!(array_member.as_referent(), <&ArrayType>::try_from(&array_member).ok());
    }

    #[test]
    fn test_array_ir_type_refinements_from_local_declarations() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, p, q, k] = ["n", "m", "p", "q", "k"].map(|name| DimensionVariable::new(name, bounds));
        let dimension = |variable: &DimensionVariable| ArrayIrType::Dimension(DimensionType::from(variable.clone()));
        let exact = |extent: usize| ArrayIrType::Dimension(DimensionType::from(Dimension::Static(extent)));
        let vector = |dimension: Dimension| ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![dimension])));
        let from_local_declarations = |declared: &[ArrayIrType], actual: &[ArrayIrType]| {
            ArrayIrTypeRefinements::from_local_declarations(declared, actual).unwrap()
        };

        // Swapped variables keep separate declared and actual facts: declared `m = 2` fixes the supplied `n` matched
        // with it, while the supplied `m` matched with declared `n` stays symbolic, regardless of member order.
        let declared = [dimension(&n), dimension(&m), vector(m.clone().into())];
        let actual = [dimension(&m), dimension(&n), vector(Dimension::Static(2))];
        for permutation in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
            let refinements = from_local_declarations(
                &permutation.map(|index| declared[index].clone()),
                &permutation.map(|index| actual[index].clone()),
            );
            assert_eq!(refinements.refine_actual(&dimension(&m), &[]), Ok(dimension(&m)));
            assert_eq!(refinements.refine_actual(&dimension(&n), &[]), Ok(exact(2)));
            assert_eq!(refinements.refine_declared(&dimension(&n), &[]), Ok(dimension(&n)));
            assert_eq!(refinements.refine_declared(&vector(m.clone().into()), &[]), Ok(vector(Dimension::Static(2))));
        }

        // A repeated supplied variable propagates a declared fact to every declared variable matched with it,
        // regardless of member order.
        let declared = [dimension(&n), dimension(&p), vector(n.clone().into())];
        let actual = [dimension(&m), dimension(&m), vector(Dimension::Static(2))];
        for permutation in [[0, 1, 2], [2, 1, 0], [1, 2, 0]] {
            let refinements = from_local_declarations(
                &permutation.map(|index| declared[index].clone()),
                &permutation.map(|index| actual[index].clone()),
            );
            assert_eq!(refinements.refine_actual(&dimension(&m), &[]), Ok(exact(2)));
            assert_eq!(refinements.refine_declared(&vector(p.clone().into()), &[]), Ok(vector(Dimension::Static(2))));
        }

        // Connected classes share facts through transitive chains and cycles of correspondences.
        let refinements = from_local_declarations(
            &[dimension(&q), dimension(&p), dimension(&p), dimension(&n), vector(n.clone().into())],
            &[dimension(&k), dimension(&k), dimension(&m), dimension(&m), vector(Dimension::Static(3))],
        );
        assert_eq!(refinements.refine_actual(&dimension(&k), &[]), Ok(exact(3)));
        assert_eq!(refinements.refine_declared(&vector(q.clone().into()), &[]), Ok(vector(Dimension::Static(3))));
        let refinements = from_local_declarations(
            &[dimension(&n), dimension(&p), dimension(&p), dimension(&n), vector(n.clone().into())],
            &[dimension(&m), dimension(&m), dimension(&k), dimension(&k), vector(Dimension::Static(2))],
        );
        assert_eq!(refinements.refine_actual(&dimension(&k), &[]), Ok(exact(2)));
        assert_eq!(refinements.refine_declared(&vector(p.clone().into()), &[]), Ok(vector(Dimension::Static(2))));

        // A static declared extent proves the extent of an actual identity with singleton bounds, and an identity with
        // singleton bounds that appears on both sides is exact in both scopes.
        let singleton = DimensionVariable::new("singleton", DimensionBounds::new(2, Some(3)).unwrap());
        let refinements = from_local_declarations(&[vector(Dimension::Static(2))], &[vector(singleton.clone().into())]);
        assert_eq!(refinements.refine_actual(&vector(singleton.clone().into()), &[]), Ok(vector(Dimension::Static(2))));
        let refinements =
            from_local_declarations(&[vector(singleton.clone().into())], &[vector(singleton.clone().into())]);
        assert_eq!(refinements.refine_actual(&vector(singleton.clone().into()), &[]), Ok(vector(Dimension::Static(2))));
        assert_eq!(
            refinements.refine_declared(&vector(singleton.clone().into()), &[]),
            Ok(vector(Dimension::Static(2))),
        );

        // A propagated fact must lie within the bounds of the identity that receives it, and propagated facts must
        // agree with the facts observed directly.
        let narrow = DimensionVariable::new("narrow", DimensionBounds::positive(Some(4)).unwrap());
        assert_eq!(
            ArrayIrTypeRefinements::from_local_declarations(
                &[dimension(&n), dimension(&p), vector(n.clone().into())],
                &[dimension(&narrow), dimension(&narrow), vector(Dimension::Static(5))],
            ),
            Err(TypeError::from(DimensionError::BindingOutOfBounds {
                variable: narrow.to_string(),
                value: 5,
                bounds: narrow.bounds(),
            })),
        );
        assert_eq!(
            ArrayIrTypeRefinements::from_local_declarations(
                &[dimension(&n), dimension(&p), vector(n.clone().into()), vector(p.clone().into())],
                &[dimension(&m), dimension(&m), vector(Dimension::Static(2)), vector(Dimension::Static(3))],
            ),
            Err(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: p.to_string(),
                expected: 3,
                actual: 2,
            })),
        );

        // A repeated declared identity entered by distinct symbolic actual identities is admitted without a fact,
        // since their equality still requires a runtime input check.
        let refinements = from_local_declarations(&[dimension(&n), dimension(&n)], &[dimension(&m), dimension(&k)]);
        assert_eq!(refinements.refine_actual(&dimension(&m), &[]), Ok(dimension(&m)));
        assert_eq!(refinements.refine_declared(&dimension(&n), &[]), Ok(dimension(&n)));

        // Nominal admission rejects a distinct array-axis variable. Local declarations record the relationship,
        // whose supplied representation can prove it without any runtime check.
        assert!(ArrayIrTypeRefinements::new(&[vector(n.clone().into())], &[vector(m.clone().into())]).is_err());
        let renamed =
            ArrayIrTypeRefinements::from_local_declarations(&[vector(n.clone().into())], &[vector(m.clone().into())])
                .unwrap();
        assert!(renamed.identities_needing_runtime_checks().is_empty());
        let scoped = ArrayIrTypeRefinements::from_local_declarations(
            &[vector(n.clone().into()), vector(n.clone().into())],
            &[vector(m.clone().into()), vector(Dimension::Static(2))],
        )
        .unwrap();
        assert_eq!(scoped.identities_needing_runtime_checks(), &[n.clone()]);
        assert_eq!(scoped.refine_actual(&vector(m.clone().into()), &[]), Ok(vector(Dimension::Static(2))));
        let wide = DimensionVariable::new("wide", DimensionBounds::positive(Some(16)).unwrap());
        assert!(
            ArrayIrTypeRefinements::from_local_declarations(&[vector(n.clone().into())], &[vector(wide.into())])
                .is_err()
        );

        // Recording a correspondence does not replace the pairwise admissibility checks.
        let declared = DimensionType::new("declared", DimensionBounds::positive(Some(4)).unwrap());
        let wider = DimensionType::new("wider", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(
            ArrayIrTypeRefinements::from_local_declarations(
                &[ArrayIrType::Dimension(declared.clone())],
                &[ArrayIrType::Dimension(wider.clone())],
            ),
            Err(TypeError::invalid(format!("type {wider} does not refine declared type {declared}"))),
        );
    }

    #[test]
    fn test_array_ir_type_refinements_identity_classes() {
        let first = DimensionVariable::new("first", DimensionBounds::non_negative(Some(8)).unwrap());
        let second = DimensionVariable::new("second", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension = |identity: &DimensionVariable| ArrayIrType::Dimension(DimensionType::from(identity.clone()));
        let vector = |dimension| ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![dimension])));
        let declared = [dimension(&first), dimension(&second), vector(second.clone().into())];
        let actual = [dimension(&second), dimension(&first), vector(Dimension::Static(2))];
        let refinements = ArrayIrTypeRefinements::from_local_declarations(&declared, &actual).unwrap();

        // Swapping the same identity objects does not connect their separate declared and actual nodes.
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![
                TypeIdentityClass::new(std::slice::from_ref(&first), std::slice::from_ref(&second), false),
                TypeIdentityClass::new(std::slice::from_ref(&second), std::slice::from_ref(&first), true),
            ],
        );
        assert_eq!(refinements.refine_actual(&actual[0], &[]), Ok(actual[0].clone()));
        assert_eq!(
            refinements.refine_actual(&actual[1], &[]),
            Ok(ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)))),
        );
    }

    #[test]
    fn test_array_ir_type_refinements_validate_refinement() {
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, p] = ["n", "m", "p"].map(|name| DimensionVariable::new(name, bounds));
        let dimension = |variable: &DimensionVariable| ArrayIrType::Dimension(DimensionType::from(variable.clone()));
        let vector = |dimension: Dimension| ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![dimension])));
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            &[dimension(&n), dimension(&p), vector(n.clone().into())],
            &[dimension(&m), dimension(&m), vector(Dimension::Static(2))],
        )
        .unwrap();

        // Validation checks outputs only against declared facts, including those implied through actual identities,
        // and an actual-only fact cannot justify an output identity that is foreign to the boundary.
        let closed = [n.clone(), p.clone()];
        assert_eq!(
            refinements.validate_refinement(&[vector(p.clone().into())], &[vector(Dimension::Static(2))], &closed),
            Ok(())
        );
        assert_eq!(
            refinements.validate_refinement(&[vector(p.clone().into())], &[vector(Dimension::Static(3))], &closed),
            Err(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: p.to_string(),
                expected: 2,
                actual: 3,
            })),
        );
        assert_eq!(
            refinements.validate_refinement(&[vector(m.clone().into())], &[vector(Dimension::Static(2))], &closed),
            Err(TypeError::invalid(format!(
                "dimension identity {m} does not belong to the validated boundary signature",
            ))),
        );

        // Identities that must stay symbolic (e.g., changing carries) keep their identity despite declared facts.
        assert_eq!(
            refinements.refine_declared(&vector(p.clone().into()), std::slice::from_ref(&p)),
            Ok(vector(p.into())),
        );
    }

    #[test]
    fn test_array_ir_type_refinements_refine_declared() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let extent = DimensionVariable::new("extent", DimensionBounds::non_negative(Some(8)).unwrap());
        let vector = |dimension: Dimension| ArrayType::new(F32, Shape::new(vec![dimension]));
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            &[ArrayIrType::Dimension(extent.clone().into()), ArrayIrType::Array(vector(rows.clone().into()))],
            &[ArrayIrType::Dimension(extent.clone().into()), ArrayIrType::Array(vector(Dimension::Static(3)))],
        )
        .unwrap();

        // Arrays and reference referents are refined through the shared array facts, while a first-class dimension
        // type defines its identity and is returned unchanged.
        let matrix = ArrayType::new(F32, Shape::new(vec![rows.clone().into(), extent.clone().into()]));
        assert_eq!(
            refinements.refine_declared(&ArrayIrType::Array(matrix), &[]),
            Ok(ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(3), extent.clone().into()])))),
        );
        assert_eq!(
            refinements.refine_declared(&ArrayIrType::Reference(ReferenceType::new(vector(rows.clone().into()))), &[]),
            Ok(ArrayIrType::Reference(ReferenceType::new(vector(Dimension::Static(3))))),
        );
        assert_eq!(
            refinements.refine_declared(&ArrayIrType::Dimension(extent.clone().into()), &[]),
            Ok(ArrayIrType::Dimension(extent.into())),
        );
        assert_eq!(
            refinements.refine_declared(&ArrayIrType::Array(vector(rows.clone().into())), std::slice::from_ref(&rows)),
            Ok(ArrayIrType::Array(vector(rows.into()))),
        );
    }

    #[test]
    fn test_array_ir_type_refinements_refine_actual() {
        let declared =
            ArrayIrType::Dimension(DimensionType::new("declared", DimensionBounds::positive(Some(8)).unwrap()));
        let actual = ArrayIrType::Dimension(DimensionType::new("actual", DimensionBounds::positive(Some(4)).unwrap()));
        let ArrayIrType::Dimension(dimension) = &declared else { unreachable!() };
        let array = ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![dimension.dimension().clone()])));
        let concrete = ArrayIrType::Array(ArrayType::new_static(F32, vec![2]));
        let refinements =
            ArrayIrTypeRefinements::from_local_declarations([&declared, &array], [&actual, &concrete]).unwrap();
        assert_eq!(
            refinements.refine_actual(&actual, &[]),
            Ok(ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)))),
        );

        // Preserved identities keep their names; other identities in the same type are still refined.
        let ArrayIrType::Dimension(actual_dimension) = &actual else { unreachable!() };
        let declared_other: ArrayIrType =
            DimensionType::new("declared_other", DimensionBounds::positive(Some(8)).unwrap()).into();
        let other = DimensionType::new("other", DimensionBounds::positive(Some(8)).unwrap());
        let exact = |extent| ArrayIrType::Dimension(DimensionType::from(Dimension::Static(extent)));
        let matrix = |dimensions: Vec<Dimension>| ArrayIrType::Array(ArrayType::new(F32, Shape::new(dimensions)));
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            [&declared, &declared, &declared_other, &declared_other],
            [&actual, &exact(2), &other.clone().into(), &exact(3)],
        )
        .unwrap();
        let actual_matrix = matrix(vec![actual_dimension.simplified_dimension(), other.simplified_dimension()]);
        assert_eq!(
            refinements.refine_actual(&actual_matrix, &[actual_dimension.to_variable()]),
            Ok(matrix(vec![actual_dimension.simplified_dimension(), Dimension::Static(3)])),
        );
        assert_eq!(
            refinements.refine_actual(&actual_matrix, &[]),
            Ok(matrix(vec![Dimension::Static(2), Dimension::Static(3)])),
        );

        // An actual identity that the invocation never related to a declared fact keeps its identity and bounds.
        let unrelated =
            ArrayIrType::Dimension(DimensionType::new("unrelated", DimensionBounds::new(3, Some(4)).unwrap()));
        assert_eq!(refinements.refine_actual(&unrelated, &[]), Ok(unrelated));

        // With no exact declared fact, the actual dimension keeps its independent identity and narrower bounds.
        let symbolic = ArrayIrTypeRefinements::from_local_declarations([&declared], [&actual]).unwrap();
        assert_eq!(symbolic.refine_actual(&actual, &[]), Ok(actual));
    }

    #[test]
    fn test_array_ir_type_refinements_retype_actual() {
        // `[f32[n], f32[n], f32[n], dimension<n>]` entered with `[f32[m], f32[3], f32[k], dimension<3>]` fixes `n`,
        // `m`, and `k` to 3. Representing the inputs with `m` gives every member of that class the identity `m`,
        // including the exact array, the differently named `k`, and the first-class dimension, while refining
        // without preferred identities narrows them all.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, k] = ["n", "m", "k"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: Dimension| ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![dimension])));
        let declared = [
            vector(n.simplified_dimension()),
            vector(n.simplified_dimension()),
            vector(n.simplified_dimension()),
            ArrayIrType::Dimension(n.clone()),
        ];
        let actual = [
            vector(m.simplified_dimension()),
            vector(Dimension::Static(3)),
            vector(k.simplified_dimension()),
            ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3))),
        ];
        let refinements = ArrayIrTypeRefinements::from_local_declarations(declared.iter(), actual.iter()).unwrap();
        let preferred_identities = [m.to_variable()];
        for (declared, actual) in declared.iter().zip(&actual) {
            let expected = match actual {
                ArrayIrType::Dimension(_) => ArrayIrType::Dimension(m.clone()),
                _ => vector(m.simplified_dimension()),
            };
            assert_eq!(refinements.retype_actual(declared, actual, &preferred_identities), Ok(expected));
            let narrowed = match actual {
                ArrayIrType::Dimension(_) => ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3))),
                _ => vector(Dimension::Static(3)),
            };
            assert_eq!(refinements.retype_actual(declared, actual, &[]), Ok(narrowed));
        }

        // A class without an exact extent can also select the preferred identity for every member,
        // while refining without preferred identities keeps every member's own identity.
        let refinements = ArrayIrTypeRefinements::from_local_declarations(
            [vector(n.simplified_dimension()), vector(n.simplified_dimension())].iter(),
            [vector(m.simplified_dimension()), vector(k.simplified_dimension())].iter(),
        )
        .unwrap();
        assert_eq!(
            refinements.retype_actual(
                &vector(n.simplified_dimension()),
                &vector(k.simplified_dimension()),
                &preferred_identities
            ),
            Ok(vector(m.simplified_dimension())),
        );
        assert_eq!(
            refinements.retype_actual(&vector(n.simplified_dimension()), &vector(k.simplified_dimension()), &[]),
            Ok(vector(k.simplified_dimension())),
        );
        assert!(
            refinements
                .retype_actual(&vector(n.simplified_dimension()), &ArrayIrType::Dimension(k), &[])
                .is_err()
        );
    }

    #[test]
    fn test_array_ir_type_refinements_retype_actual_preserves_declared_bounds() {
        let wide = DimensionType::new("wide", DimensionBounds::positive(Some(8)).unwrap());
        let narrow = DimensionType::new("narrow", DimensionBounds::positive(Some(4)).unwrap());
        let exact = DimensionType::new("exact", DimensionBounds::new(3, Some(4)).unwrap());
        let reference: ArrayIrType =
            ReferenceType::new(ArrayType::new(F32, Shape::new(vec![wide.simplified_dimension()]))).into();
        let matrix = |dimensions| ArrayIrType::Array(ArrayType::new(F32, Shape::new(dimensions)));
        let declared = [
            reference.clone(),
            narrow.clone().into(),
            matrix(vec![narrow.simplified_dimension(), wide.simplified_dimension()]),
        ];
        let actual = [
            reference.clone(),
            exact.clone().into(),
            matrix(vec![exact.simplified_dimension(), exact.simplified_dimension()]),
        ];
        let refinements = ArrayIrTypeRefinements::from_local_declarations(&declared, &actual).unwrap();
        let retained = [wide.to_variable()];

        // The unchanged reference keeps wide, but that name cannot widen narrow's authoritative bounds.
        assert_eq!(refinements.retype_actual(&declared[0], &actual[0], &retained), Ok(reference));
        assert_eq!(
            refinements.retype_actual(&declared[1], &actual[1], &retained),
            Ok(ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3)))),
        );
        assert_eq!(
            refinements.retype_actual(&declared[2], &actual[2], &retained),
            Ok(matrix(vec![Dimension::Static(3), wide.simplified_dimension()])),
        );

        // A retained actual name selected by an earlier representation also yields to the narrower declaration.
        assert_eq!(
            refinements.retype_actual(&declared[1], &wide.into(), &retained),
            Ok(ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3)))),
        );
    }

    #[test]
    fn test_array_ir_type_refinements_retype_actual_unfixed_compatible_representations() {
        let wide = DimensionType::new("wide", DimensionBounds::positive(Some(8)).unwrap());
        let narrow = DimensionType::new("narrow", DimensionBounds::positive(Some(4)).unwrap());
        let actual_dimension = DimensionType::new("actual", narrow.bounds());
        let reference: ArrayIrType =
            ReferenceType::new(ArrayType::new(F32, Shape::new(vec![wide.simplified_dimension()]))).into();
        let vector: ArrayIrType = ArrayType::new(F32, Shape::new(vec![narrow.simplified_dimension()])).into();
        let declared = [reference.clone(), narrow.clone().into(), wide.clone().into(), vector.clone()];
        let actual =
            [reference.clone(), actual_dimension.clone().into(), actual_dimension.clone().into(), vector.clone()];
        let retained = [wide.to_variable()];
        let refinements = ArrayIrTypeRefinements::from_local_declarations(&declared, &actual).unwrap();
        assert!(!refinements.identity_classes().next().unwrap().is_exact());

        // The wide reference's protected name cannot represent the narrower body component. Both its dimension and
        // its vector keep the compatible boundary identity, while wide's dimension follows the reference's name.
        assert_eq!(refinements.retype_actual(&declared[0], &actual[0], &retained), Ok(reference.clone()));
        assert_eq!(refinements.retype_actual(&declared[1], &actual[1], &retained), Ok(narrow.clone().into()));
        assert_eq!(refinements.retype_actual(&declared[2], &actual[2], &retained), Ok(wide.clone().into()));
        assert_eq!(refinements.retype_actual(&declared[3], &actual[3], &retained), Ok(vector));

        // Without a carried boundary identity, the first compatible actual member represents the narrower component.
        let refinements = ArrayIrTypeRefinements::from_local_declarations(&declared[..3], &actual[..3]).unwrap();
        assert_eq!(refinements.retype_actual(&declared[1], &actual[1], &retained), Ok(actual_dimension.clone().into()));

        // Representation normalization is requested only for a class with a protected member.
        assert_eq!(refinements.retype_actual(&declared[1], &actual[1], &[]), Ok(actual_dimension.into()));
    }
}
