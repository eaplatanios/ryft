use std::borrow::Borrow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::types::arrays::{ArrayType, ArrayTypeRefinements};
use crate::arrays::types::dimensions::{Dimension, DimensionType, DimensionVariable};
use crate::parameters::Parameter;
use crate::programs::types::visit_type_signature_pairs;
use crate::programs::{ReferenceType, Type, TypeError, TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements};

/// [`Type`] vocabulary of Ryft's array Intermediate Representation (IR), whose values may be ordinary arrays,
/// first-class runtime dimensions, or [`Reference`](crate::Reference)s to arrays. It is the type-level counterpart of
/// [`ArrayIrValue`](crate::ArrayIrValue), with one member per value kind.
///
/// The sum is a storage boundary rather than the contract ordinary primitives are written against. Array-only
/// [`Operation`](crate::Operation)s and transform rules keep consuming [`ArrayType`], dimension-only operations keep
/// consuming [`DimensionType`], and genuinely mixed operations consume this type directly. Those mixed operations
/// include shape-carrying operations with explicit dimension operands and reference operations whose signatures cross
/// the array/reference boundary. [`From`] lifts each member type into the sum, and the borrowing [`TryFrom`]
/// implementations project it back out with a checked kind diagnostic. The same bridge backs the value-level
/// [`ValueProjection`](crate::ValueProjection) implementations.
///
/// All three members can carry [`DimensionVariable`] identities, so one renaming and refinement vocabulary spans a
/// complete signature. An [`ArrayType`] member and the referent of a [`ReferenceType`] member _reference_ the variables
/// named by their dynamic axes, while a [`DimensionType`] member _defines_ its variable.
/// [`Type::derive_identity_renaming`] therefore checks a variable repeated across member kinds for consistency, exactly
/// as it does within one member kind.
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
    /// Projects a run of explicit extent operand types into the [`Dimension`]s they define, in operand order.
    ///
    /// Mixed shape-carrying operations derive their result shape from exactly this projection, so every mixed
    /// inference rule that consumes a trailing extent-operand run shares one member-kind diagnostic for an operand
    /// that is not a dimension.
    ///
    /// # Parameters
    ///
    ///   - `types`: Extent operand types, in operand order. The [`Borrow`] item bound lets type slices, owned [`Type`]
    ///     iterators, and borrowed [`Type`]s all be projected in place, so value-level callers need no intermediate
    ///     type collection.
    pub(crate) fn extents<T: IntoIterator<Item: Borrow<Self>>>(types: T) -> Result<Vec<Dimension>, TypeError> {
        types
            .into_iter()
            .map(|r#type| <&DimensionType>::try_from(r#type.borrow()).map(DimensionType::to_dimension))
            .collect()
    }

    /// Returns this composite member's diagnostic kind name.
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

    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        let array = match self {
            Self::Array(r#type) => Some(r#type),
            Self::Dimension(_) | Self::Reference(_) => None,
        };
        let dimension = match self {
            Self::Array(_) | Self::Reference(_) => None,
            Self::Dimension(r#type) => Some(r#type),
        };
        let reference = match self {
            Self::Array(_) | Self::Dimension(_) => None,
            Self::Reference(r#type) => Some(r#type),
        };
        array
            .into_iter()
            .flat_map(ArrayType::identities)
            .chain(dimension.into_iter().flat_map(DimensionType::identities))
            .chain(reference.into_iter().flat_map(ReferenceType::identities))
    }

    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
        let mut renaming = TypeIdentityRenaming::new();
        let mut refinements = ArrayTypeRefinements::default();
        visit_type_signature_pairs(declared, actual, |declared, actual| match (declared, actual) {
            (Self::Array(declared), Self::Array(actual)) => {
                ArrayType::extend_identity_renaming(declared, actual, &mut renaming, &mut refinements)
            }
            (Self::Dimension(declared), Self::Dimension(actual)) => {
                DimensionType::extend_identity_renaming(declared, actual, &mut renaming)
            }
            (Self::Reference(declared), Self::Reference(actual)) => ArrayType::extend_identity_renaming(
                declared.referent(),
                actual.referent(),
                &mut renaming,
                &mut refinements,
            ),
            (Self::Array(_), actual) => {
                Err(TypeError::invalid(format!("expected array type but got {} type", actual.kind_name())))
            }
            (Self::Dimension(_), actual) => {
                Err(TypeError::invalid(format!("expected dimension type but got {} type", actual.kind_name())))
            }
            (Self::Reference(_), actual) => {
                Err(TypeError::invalid(format!("expected reference type but got {} type", actual.kind_name())))
            }
        })?;
        refinements.require_disjoint_from(&renaming)?;
        Ok(renaming)
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

    #[inline]
    fn referent(&self) -> Option<Self> {
        match self {
            Self::Reference(r#type) => Some(Self::Array(r#type.referent().clone())),
            Self::Array(_) | Self::Dimension(_) => None,
        }
    }
}

/// [`TypeRefinements`] established while refining one complete [`ArrayIrType`] signature. A declared dynamic array
/// axis met by a static extent contributes a dynamic-to-static binding following the [`ArrayTypeRefinements`] rules
/// unchanged. A dimension member contributes no concrete fact, because a [`DimensionType`] is strictly identity
/// plus bounds. Its variable belongs to the boundary's closed identity set (i.e., for more information refer to
/// [`TypeIdentitySignature`](crate::TypeIdentitySignature)), which lets output validation establish the concrete
/// extent on first observation (e.g., relating an eagerly materialized static array output back to the dimension
/// input that supplied its shape) while still rejecting inconsistent repeated observations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayIrTypeRefinements {
    /// [`ArrayTypeRefinements`] shared by array members and reference referents across the complete signature.
    arrays: ArrayTypeRefinements,
}

impl ArrayIrTypeRefinements {
    /// Validates that `actual` refines `declared` and visits any concrete identity refinement the pair contributes,
    /// dispatching on the composite member kind.
    ///
    /// This is the composite counterpart of [`ArrayTypeRefinements::visit_dynamic_to_static_refinements`] and the
    /// shared per-type engine behind this family's [`TypeRefinements::establish`] and [`TypeRefinements::validate`].
    /// The member dispatch lives here exactly once, while the provided `visit` callback decides what an observed
    /// concrete extent means (binding a new fact versus justifying an observation against previously established
    /// facts). The member kinds behave differently:
    ///
    ///   - An array pair delegates to [`ArrayTypeRefinements::visit_dynamic_to_static_refinements`], so declared
    ///     dynamic axes met by in-bounds static extents reach `visit` under the ordinary array refinement rules.
    ///   - A dimension pair only checks that `actual` refines `declared` (same [`DimensionVariable`], equal or
    ///     narrowed bounds) and never calls `visit`: a [`DimensionType`] is strictly identity plus bounds, so the
    ///     pair carries no concrete extent to record. Concrete facts about such an identity can only be established
    ///     later, when output validation first observes it on an array member (refer to the [`ArrayIrTypeRefinements`]
    ///     documentation for how the closed identity set makes that sound).
    ///   - A reference pair applies the array rules to its referents using the same refinement accumulator.
    ///   - Mismatched member kinds fail, since values never refine across composite member kinds.
    ///
    /// # Parameters
    ///
    ///   - `declared`: [`ArrayIrType`] declared by the boundary signature.
    ///   - `actual`: Observed [`ArrayIrType`] that must refine `declared`, including having the same member kind.
    ///   - `visit`: Callback invoked once per in-bounds dynamic-to-static array axis, in axis order, with the
    ///     declared [`DimensionVariable`] and the observed static extent. Any error it returns aborts the walk and
    ///     propagates to the caller.
    fn visit_dynamic_to_static_refinements(
        declared: &ArrayIrType,
        actual: &ArrayIrType,
        visit: impl FnMut(&DimensionVariable, usize) -> Result<(), TypeError>,
    ) -> Result<(), TypeError> {
        match (declared, actual) {
            (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) => {
                ArrayTypeRefinements::visit_dynamic_to_static_refinements(declared, actual, visit)
            }
            (ArrayIrType::Dimension(declared), ArrayIrType::Dimension(actual)) if declared.is_refined_by(actual) => {
                Ok(())
            }
            (ArrayIrType::Dimension(declared), ArrayIrType::Dimension(actual)) => {
                Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")))
            }
            (ArrayIrType::Reference(declared), ArrayIrType::Reference(actual)) => {
                ArrayTypeRefinements::visit_dynamic_to_static_refinements(declared.referent(), actual.referent(), visit)
            }
            // These cross-kind arms below stay keyed on the declared variant so that adding a composite member forces a
            // compile-time decision here (a full catch-all would silently treat a new same-kind pair as a mismatch).
            (ArrayIrType::Array(_), actual) => {
                Err(TypeError::invalid(format!("expected array type but got {} type", actual.kind_name())))
            }
            (ArrayIrType::Dimension(_), actual) => {
                Err(TypeError::invalid(format!("expected dimension type but got {} type", actual.kind_name())))
            }
            (ArrayIrType::Reference(_), actual) => {
                Err(TypeError::invalid(format!("expected reference type but got {} type", actual.kind_name())))
            }
        }
    }
}

impl TypeRefinements<ArrayIrType> for ArrayIrTypeRefinements {
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayIrType>,
        A::Item: Borrow<ArrayIrType>,
    {
        let mut refinements = Self::default();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::visit_dynamic_to_static_refinements(declared, actual, |variable, extent| {
                refinements.arrays.bind(variable, extent)
            })
        })?;
        Ok(refinements)
    }

    fn validate<D: IntoIterator, A: IntoIterator>(
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
        let mut refinements = self.clone();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::visit_dynamic_to_static_refinements(declared, actual, |variable, extent| {
                refinements.arrays.validate_or_bind(variable, extent, closed_identities)
            })
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

        // Extent projection accepts an empty run, preserves operand order, recognizes singleton-bounded dimensions as
        // static, retains wider-bounded dimensions as dynamic, and rejects non-dimension members.
        let exact_variable = DimensionVariable::new("exact", DimensionBounds::new(4, Some(5)).unwrap());
        let extent_types = [
            ArrayIrType::Dimension(DimensionType::new(exact_variable)),
            ArrayIrType::Dimension(DimensionType::new(declared_variable.clone())),
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
            ArrayIrType::Dimension(DimensionType::new(declared_variable.clone())),
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(declared_variable.clone())]))),
        ];
        let actual = [
            ArrayIrType::Dimension(DimensionType::new(actual_variable.clone())),
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
            ArrayIrType::derive_identity_renaming(&declared[..1], &[ArrayIrType::Array(ArrayType::scalar(F32))],),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        // Repeated observations of one dynamic array identity establish one concrete refinement and reject a later
        // conflicting extent.
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared_array =
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(batch.clone())])));
        let actual_two = ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])));
        let actual_three = ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)])));
        let refinements = ArrayIrTypeRefinements::establish(
            [declared_array.clone(), declared_array.clone()],
            [actual_two.clone(), actual_two.clone()],
        )
        .unwrap();
        assert_eq!(refinements.validate([declared_array.clone()], [actual_two.clone()], &[]), Ok(()));
        let error = ArrayIrTypeRefinements::establish(
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
        let mixed_refinements = ArrayIrTypeRefinements::establish(
            [declared_array.clone(), declared_reference.clone()],
            [actual_two.clone(), actual_reference_two],
        )
        .unwrap();
        assert_eq!(
            mixed_refinements.validate(
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
        let error = ArrayIrTypeRefinements::establish(
            [declared_array.clone(), declared_reference],
            [actual_two.clone(), actual_reference_three],
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        // A dimension member contributes no concrete fact (its type is strictly identity plus bounds). Its variable
        // instead belongs to the boundary's closed identity signature, so output validation may establish the concrete
        // extent for it on first observation and must reject an inconsistent repeated observation within the same
        // validated signature.
        let declared_dimension = ArrayIrType::Dimension(DimensionType::new(batch.clone()));
        let refinements = ArrayIrTypeRefinements::establish(
            std::slice::from_ref(&declared_dimension),
            std::slice::from_ref(&declared_dimension),
        )
        .unwrap();
        let closed_identities = [batch.clone()];
        assert_eq!(
            refinements.validate(
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
            .validate(
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

        // An identity outside the boundary's closed signature stays rejected, and an internally defined identity may
        // establish its first fact exactly like an input-signature identity.
        let unrelated = DimensionVariable::new("unrelated", DimensionBounds::non_negative(Some(8)).unwrap());
        let unrelated_array =
            ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(unrelated.clone())])));
        assert_eq!(
            refinements.validate(
                std::slice::from_ref(&unrelated_array),
                [ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])))],
                &closed_identities,
            ),
            Err(TypeError::invalid("dimension identity unrelated does not belong to the validated boundary signature")),
        );
        assert_eq!(
            refinements.validate(
                std::slice::from_ref(&unrelated_array),
                [ArrayIrType::Array(ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])))],
                std::slice::from_ref(&unrelated),
            ),
            Ok(()),
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
        let dimension =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap()));
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
        assert_eq!(stored.referent(), Some(ArrayIrType::Array(array.clone())));
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
}
