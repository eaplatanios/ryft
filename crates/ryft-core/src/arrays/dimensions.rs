use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};

use ryft_macros::Parameter;

use crate::arrays::operations::DimensionOperation;
use crate::arrays::types::dimensions::{Dimension, DimensionError, DimensionType, MAX_DIMENSION_EXTENT};
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::{
    Concretizable, LiteralIdentity, ProgramError, Type, TypeError, TypeIdentityRenaming, Typed, Value,
    ValueDirectDispatch,
};

/// Checked host representation of a first-class runtime [`Dimension`] value. Its eager domain performs checked host
/// integer arithmetic without allocating an array or dispatching to a device backend. Fallible capabilities such as
/// [`Add::add`](crate::Add::add) form the canonical arithmetic API. [`std::ops::Add`], [`std::ops::Sub`],
/// [`std::ops::Mul`], [`std::ops::Div`], and [`std::ops::Rem`] implementations provide panicking operator
/// sugar for both owned and borrowed values.
///
/// Like every eager value, a [`DimensionValue`] is typed by its exact type: the static [`DimensionType`] of its extent
/// (rendered as, e.g., `dimension<3>`), just as an eager array is typed by its static shape. A static type has no
/// identity, so two values are equal exactly when their types are (i.e., when their extents are, like the host integers
/// that they represent). Symbolic dimension types, whose bounds admit a range of extents, describe staged values
/// instead. A Just-In-Time (JIT) compilation boundary that should reuse one compiled program across extents declares
/// the symbolic types at which to stage its inputs explicitly. Refer to the documentation of
/// [`CompiledFunctionDispatcher::with_staged_input_types`](crate::CompiledFunctionDispatcher::with_staged_input_types)
/// for more information on that.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionValue {
    /// Concrete non-negative extent of this [`DimensionValue`].
    extent: usize,
}

impl DimensionValue {
    /// Creates a new [`DimensionValue`] with the provided `extent` after validating it against the bounds of the
    /// provided [`DimensionType`] (e.g., the declared type of the program input that the value is meant for). The
    /// value's own type is the static type of `extent` regardless of `r#type`.
    pub fn new(r#type: DimensionType, extent: usize) -> Result<Self, DimensionError> {
        let bounds = r#type.bounds();
        if !bounds.contains(extent) {
            return Err(DimensionError::BindingOutOfBounds {
                variable: r#type.simplified_dimension().to_string(),
                value: extent,
                bounds,
            });
        }
        Self::constant(extent)
    }

    /// Creates a new [`DimensionValue`] with the provided `extent`.
    #[inline]
    pub fn constant(extent: usize) -> Result<Self, DimensionError> {
        if extent > MAX_DIMENSION_EXTENT {
            return Err(DimensionError::ExtentExceedsBackendWidth { value: extent, maximum: MAX_DIMENSION_EXTENT });
        }
        Ok(Self { extent })
    }

    /// Returns the concrete non-negative extent of this [`DimensionValue`].
    #[inline]
    pub fn extent(&self) -> usize {
        self.extent
    }
}

impl Display for DimensionValue {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.extent, formatter)
    }
}

impl Typed for DimensionValue {
    type Type = DimensionType;

    #[inline]
    fn r#type(&self) -> Cow<'_, DimensionType> {
        Cow::Owned(DimensionType::from(Dimension::Static(self.extent)))
    }
}

impl Value for DimensionValue {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self, DimensionOperation<Self>>;

    #[inline]
    fn domain(&self) -> Self::Domain {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        // A static type has no identities to rename.
        Ok(self.clone())
    }

    #[inline]
    fn singleton(r#type: &DimensionType) -> Option<Self> {
        // Bounds are not checked against the backend width when they are constructed, so a singleton interval above
        // `MAX_DIMENSION_EXTENT` admits one extent that no value can represent; it has no singleton value.
        r#type.extent().and_then(|extent| Self::constant(extent).ok())
    }
}

impl LiteralIdentity for DimensionValue {
    // A static type has no identity, so the extent identifies a dimension literal completely.

    #[inline]
    fn literal_eq(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn literal_hash<H: Hasher>(&self, state: &mut H) {
        self.hash(state);
    }
}

impl Concretizable<usize> for DimensionValue {
    #[inline]
    fn concretize(&self) -> Result<usize, ProgramError> {
        Ok(self.extent)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::dimensions::DimensionBounds;
    use crate::operations::{Add, Assert, Compare, Div, Rem, Sub};

    use crate::tests::literal_hash_of;

    use super::*;

    #[test]
    fn test_dimension_value() {
        let batch_type = DimensionType::new("batch", DimensionBounds::new(1, Some(65)).unwrap());
        let batch = DimensionValue::new(batch_type.clone(), 32).unwrap();

        // The value's type is the static type of its extent, which has no identity, so every value of that extent has
        // the same type regardless of the type that it was validated against.
        assert_eq!(batch.r#type().into_owned(), DimensionType::from(Dimension::Static(32)));
        assert_eq!(batch.r#type().variable(), None);
        assert_eq!(batch.r#type().to_string(), "dimension<32>");
        assert_eq!(batch.r#type(), DimensionValue::constant(32).unwrap().r#type());
        assert_eq!(batch.extent(), 32);
        assert_eq!(batch.to_string(), "32");
        assert_eq!(batch.concretize(), Ok(32));
        assert_eq!(
            DimensionValue::new(batch_type, 65),
            Err(DimensionError::BindingOutOfBounds {
                variable: "batch".to_string(),
                value: 65,
                bounds: DimensionBounds::new(1, Some(65)).unwrap(),
            }),
        );
        if let Some(unsupported_extent) = MAX_DIMENSION_EXTENT.checked_add(1) {
            assert_eq!(
                DimensionValue::constant(unsupported_extent),
                Err(DimensionError::ExtentExceedsBackendWidth {
                    value: unsupported_extent,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }

        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();
        assert_eq!(left.add(&right).unwrap().extent(), 10);
        let predicate: Array = left.less_than_or_equal(&DimensionValue::constant(8).unwrap()).unwrap();
        predicate.assert("dimension is at most `8`", &[]).unwrap();

        // Concrete capabilities derive fresh result types from their inputs without exposing IR operations in the
        // value-level API.
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(10)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(0, Some(10)).unwrap());
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().extent(), Some(10));
        assert_eq!(sum.extent(), 10);

        // Concrete capability errors report operand roles and actual extents, independently of the static types.
        let error = DimensionValue::new(left_type.clone(), 1).unwrap().sub(&right).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "`dimension_sub` requires left >= right; got left=1, right=3".to_string(),
            }),
        );
        let zero = DimensionValue::new(right_type.clone(), 0).unwrap();
        let error = left.div(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "`dimension_div` requires right > 0; got left=7, right=0".to_string(),
            }),
        );
        let error = left.rem(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "`dimension_rem` requires right > 0; got left=7, right=0".to_string(),
            }),
        );
    }

    #[test]
    fn test_dimension_value_singleton() {
        // A singleton interval determines its value, whose type is static like that of every dimension value.
        let singleton_type = DimensionType::new("batch", DimensionBounds::new(4, Some(5)).unwrap());
        let singleton = DimensionValue::singleton(&singleton_type).unwrap();
        assert_eq!(singleton, DimensionValue::constant(4).unwrap());
        assert_eq!(DimensionValue::singleton(&DimensionValue::constant(0).unwrap().r#type()).unwrap().extent(), 0);

        // Wider bounds admit several extents and so determine no value.
        let wide_type = DimensionType::new("batch", DimensionBounds::new(4, Some(6)).unwrap());
        assert_eq!(DimensionValue::singleton(&wide_type), None);
        assert_eq!(DimensionValue::singleton(&DimensionType::new("batch", DimensionBounds::unbounded())), None);

        // A singleton interval above the backend width admits one extent that no value can represent.
        if let Some(unsupported_extent) = MAX_DIMENSION_EXTENT.checked_add(1)
            && let Some(unsupported_upper) = unsupported_extent.checked_add(1)
        {
            let unsupported_bounds = DimensionBounds::new(unsupported_extent, Some(unsupported_upper)).unwrap();
            assert_eq!(DimensionValue::singleton(&DimensionType::new("batch", unsupported_bounds)), None);
        }
    }

    #[test]
    fn test_literal_identity_dimension_value() {
        // Dimension literals have static types without identities, so they are identical exactly
        // when their extents are.
        let dimension = DimensionValue::constant(3).unwrap();
        assert!(dimension.literal_eq(&DimensionValue::constant(3).unwrap()));
        assert_eq!(literal_hash_of(&dimension), literal_hash_of(&DimensionValue::constant(3).unwrap()));
        assert!(!dimension.literal_eq(&DimensionValue::constant(4).unwrap()));
    }
}
