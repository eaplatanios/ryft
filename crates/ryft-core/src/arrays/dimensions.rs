use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::operations::DimensionOperation;
use crate::arrays::types::dimensions::{
    DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT,
};
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::{Concretizable, ProgramError, Type, TypeError, TypeIdentityRenaming, Typed, Value};

/// Checked host representation of a first-class runtime [`Dimension`](crate::Dimension) value. Its eager domain
/// performs checked host integer arithmetic without allocating an array or dispatching to a device backend. Fallible
/// capabilities such as [`Add::add`](crate::Add::add) form the canonical arithmetic API. [`std::ops::Add`],
/// [`std::ops::Sub`], [`std::ops::Mul`], [`std::ops::Div`], and [`std::ops::Rem`] implementations
/// provide panicking operator sugar for both owned and borrowed values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionValue {
    /// [`DimensionType`] defining this [`DimensionValue`]'s [`DimensionVariable`] and [`DimensionBounds`].
    r#type: DimensionType,

    /// Concrete non-negative extent of this [`DimensionValue`].
    extent: usize,
}

impl DimensionValue {
    /// Creates a new [`DimensionValue`] after validating the provided [`DimensionType`] and `extent`.
    pub fn new(r#type: DimensionType, extent: usize) -> Result<Self, DimensionError> {
        if extent > MAX_DIMENSION_EXTENT {
            return Err(DimensionError::ExtentExceedsBackendWidth { value: extent, maximum: MAX_DIMENSION_EXTENT });
        }

        let bounds = r#type.bounds();
        if !bounds.contains(extent) {
            return Err(DimensionError::BindingOutOfBounds {
                variable: r#type.variable().to_string(),
                value: extent,
                bounds,
            });
        }

        Ok(Self { r#type, extent })
    }

    /// Creates a new [`DimensionValue`] literal with a fresh [`DimensionType`] that admits only `extent`.
    #[inline]
    pub fn constant(extent: usize) -> Result<Self, DimensionError> {
        let bounds = DimensionBounds::new(extent, extent.checked_add(1))?;
        Self::new(DimensionType::new(DimensionVariable::new(extent.to_string(), bounds)), extent)
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
        Cow::Borrowed(&self.r#type)
    }
}

impl Value for DimensionValue {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self, DimensionOperation<Self>>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Self::new(self.r#type.rename_identities(renaming)?, self.extent).map_err(Into::into)
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

    use crate::operations::{Add, DimensionRequirement, Div, Rem, Sub};

    use super::*;

    #[test]
    fn test_dimension_value() {
        let batch_type =
            DimensionType::new(DimensionVariable::new("batch", DimensionBounds::new(1, Some(65)).unwrap()));
        let batch = DimensionValue::new(batch_type.clone(), 32).unwrap();
        assert_eq!(batch.r#type().as_ref(), &batch_type);
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
        left.require_less_than_or_equal(&DimensionValue::constant(8).unwrap()).unwrap();

        // Concrete capabilities derive fresh result types from their operands without exposing IR operations in the
        // value-level API.
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right_type =
            DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().bounds(), DimensionBounds::new(0, Some(19)).unwrap());
        assert_eq!(sum.extent(), 10);

        // Concrete backend capability implementations retain operand-specific diagnostics for invalid observed
        // extents admitted by otherwise valid operand bounds.
        let error = DimensionValue::new(left_type.clone(), 1).unwrap().sub(&right).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left >= right; observed left=1, right=3".to_string(),
            }),
        );
        let zero = DimensionValue::new(right_type.clone(), 0).unwrap();
        let error = left.div(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
        let error = left.rem(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }
}
