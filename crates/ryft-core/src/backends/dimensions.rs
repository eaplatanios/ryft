//! Contains the host representation and closed operation family for first-class runtime dimensions.
//!
//! [`DimensionValue`] is an ordinary scalar SSA value whose [`DimensionType`] defines one
//! [`DimensionVariable`]. Arithmetic produces fresh bounded variables through nominal operations in
//! [`crate::operations::dimensions`].

use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::{Operation, Parameter};

use crate::contexts::EagerContext;
use crate::operations::constants::ConstantOperation;
use crate::operations::dimensions::{
    DimensionAdd, DimensionAddOperation, DimensionFloorDivide, DimensionFloorDivideOperation, DimensionMaximum,
    DimensionMaximumOperation, DimensionMinimum, DimensionMinimumOperation, DimensionMultiply,
    DimensionMultiplyOperation, DimensionPower, DimensionPowerOperation, DimensionRemainder,
    DimensionRemainderOperation, DimensionRequirement, DimensionRequirementOperation, DimensionSubtract,
    DimensionSubtractClamped, DimensionSubtractClampedOperation, DimensionSubtractOperation, checked_power,
    requirement_violation,
};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Concretizable, Value};
use crate::tracing::TracingContext;
use crate::types::{DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT};

// TODO(eaplatanios): Review this module.

/// [`TracingContext`] over the homogeneous dimension universe.
pub type DimensionTracingContext = TracingContext<DimensionValue, DimensionOperation<DimensionValue>>;

/// Checked host representation of one first-class runtime dimension.
///
/// Its eager domain performs checked host integer arithmetic without allocating an array or dispatching to a device
/// backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionValue {
    /// Type defining this value's dimension variable and authoritative bounds.
    r#type: DimensionType,

    /// Concrete nonnegative extent.
    extent: usize,
}

impl DimensionValue {
    /// Constructs a dimension literal with a fresh type that admits only `extent`.
    pub fn constant(extent: usize) -> Result<Self, DimensionError> {
        let bounds = DimensionBounds::new(extent, extent.checked_add(1))?;
        Self::new(DimensionType::new(DimensionVariable::new(extent.to_string(), bounds)), extent)
    }

    /// Constructs a host dimension value after validating its portable width and declared bounds.
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

    /// Returns this value's [`DimensionType`].
    #[inline]
    pub fn r#type(&self) -> &DimensionType {
        &self.r#type
    }

    /// Returns this value's concrete nonnegative extent.
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

/// Implements one concrete checked-arithmetic capability for [`DimensionValue`].
macro_rules! impl_arithmetic_dimension_capability {
    // This branch validates the supplied operation metadata, computes one host extent, and preserves its result type.
    (
        $capability:ident, $method:ident, $operation:ty,
        |$left:ident, $right:ident, $operation_argument:ident| $evaluate:expr $(,)?
    ) => {
        impl $capability for DimensionValue {
            fn $method(&self, right: &Self, $operation_argument: &$operation) -> Result<Self, ProgramError> {
                Operation::infer_output_types($operation_argument, &[self.r#type.clone(), right.r#type.clone()], &[])?;
                let $left = self.extent;
                let $right = right.extent;
                let extent: Result<usize, DimensionError> = $evaluate;
                Ok(Self::new($operation_argument.result_type(), extent?)?)
            }
        }
    };
}

impl_arithmetic_dimension_capability!(
    DimensionAdd,
    add_dimension_with,
    DimensionAddOperation,
    |left, right, operation| left.checked_add(right).ok_or_else(|| DimensionError::ArithmeticOverflow {
        message: format!(
            "dimension arithmetic overflow while adding runtime dimensions with operands {}={left}, {}={right}",
            operation.left_type().variable(),
            operation.right_type().variable(),
        ),
    }),
);
impl_arithmetic_dimension_capability!(
    DimensionSubtract,
    subtract_dimension_with,
    DimensionSubtractOperation,
    |left, right, operation| left.checked_sub(right).ok_or_else(|| requirement_violation(
        format!("{} >= {}", operation.left_type().variable(), operation.right_type().variable()),
        operation.left_type(),
        left,
        operation.right_type(),
        right,
    )),
);
impl_arithmetic_dimension_capability!(
    DimensionSubtractClamped,
    subtract_dimension_clamped_with,
    DimensionSubtractClampedOperation,
    |left, right, _operation| Ok(left.saturating_sub(right)),
);
impl_arithmetic_dimension_capability!(
    DimensionMultiply,
    multiply_dimension_with,
    DimensionMultiplyOperation,
    |left, right, operation| left.checked_mul(right).ok_or_else(|| DimensionError::ArithmeticOverflow {
        message: format!(
            "dimension arithmetic overflow while multiplying runtime dimensions with operands {}={left}, {}={right}",
            operation.left_type().variable(),
            operation.right_type().variable(),
        ),
    }),
);
impl_arithmetic_dimension_capability!(
    DimensionPower,
    raise_dimension_to_power_with,
    DimensionPowerOperation,
    |left, right, operation| checked_power(left, right).ok_or_else(|| DimensionError::ArithmeticOverflow {
        message: format!(
            "dimension arithmetic overflow while raising a runtime dimension to a dimension power with operands \
             {}={left}, {}={right}",
            operation.left_type().variable(),
            operation.right_type().variable(),
        ),
    }),
);
impl_arithmetic_dimension_capability!(
    DimensionFloorDivide,
    floor_divide_dimension_with,
    DimensionFloorDivideOperation,
    |left, right, operation| if right == 0 {
        Err(requirement_violation(
            format!("{} > 0", operation.right_type().variable()),
            operation.left_type(),
            left,
            operation.right_type(),
            right,
        ))
    } else {
        Ok(left / right)
    },
);
impl_arithmetic_dimension_capability!(
    DimensionRemainder,
    remainder_dimension_with,
    DimensionRemainderOperation,
    |left, right, operation| if right == 0 {
        Err(requirement_violation(
            format!("{} > 0", operation.right_type().variable()),
            operation.left_type(),
            left,
            operation.right_type(),
            right,
        ))
    } else {
        Ok(left % right)
    },
);
impl_arithmetic_dimension_capability!(
    DimensionMinimum,
    minimum_dimension_with,
    DimensionMinimumOperation,
    |left, right, _operation| Ok(left.min(right)),
);
impl_arithmetic_dimension_capability!(
    DimensionMaximum,
    maximum_dimension_with,
    DimensionMaximumOperation,
    |left, right, _operation| Ok(left.max(right)),
);

impl DimensionRequirement for DimensionValue {
    fn require_with(
        &self,
        right: Option<&Self>,
        operation: &DimensionRequirementOperation,
    ) -> Result<(), ProgramError> {
        match right {
            Some(right) => {
                operation.infer_output_types(&[self.r#type.clone(), right.r#type.clone()], &[])?;
            }
            None => {
                operation.infer_output_types(std::slice::from_ref(&self.r#type), &[])?;
            }
        }
        operation.evaluate_extents(self.extent, right.map(|right| right.extent))?;
        Ok(())
    }
}

/// Closed operation family stored by programs over first-class runtime dimensions.
///
/// Each arithmetic variant contains a distinct nominal operation type. This enum is the single dynamic dispatch
/// boundary needed to store heterogeneous instructions in a homogeneous [`crate::Program`]; the operation itself
/// contains no second arithmetic selector.
#[derive(Clone, Debug, Operation)]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    /// Dimension literal.
    Constant(ConstantOperation<V>),

    /// Checked dimension addition.
    Add(DimensionAddOperation),

    /// Checked dimension subtraction.
    Subtract(DimensionSubtractOperation),

    /// Clamped dimension subtraction.
    SubtractClamped(DimensionSubtractClampedOperation),

    /// Checked dimension multiplication.
    Multiply(DimensionMultiplyOperation),

    /// Checked dimension exponentiation.
    Power(DimensionPowerOperation),

    /// Checked dimension floor division.
    FloorDivide(DimensionFloorDivideOperation),

    /// Checked dimension remainder.
    Remainder(DimensionRemainderOperation),

    /// Dimension minimum.
    Minimum(DimensionMinimumOperation),

    /// Dimension maximum.
    Maximum(DimensionMaximumOperation),

    /// Ordered runtime dimension requirement.
    Requirement(DimensionRequirementOperation),
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::dimensions::{
        DimensionAdd, DimensionAddOperation, DimensionFloorDivide, DimensionFloorDivideOperation, DimensionRemainder,
        DimensionRemainderOperation, DimensionRequirement, DimensionSubtract, DimensionSubtractOperation,
    };

    use super::*;

    #[test]
    fn test_dimension_value() {
        let batch_type =
            DimensionType::new(DimensionVariable::new("batch", DimensionBounds::new(1, Some(65)).unwrap()));
        let batch = DimensionValue::new(batch_type.clone(), 32).unwrap();
        assert_eq!(batch.r#type(), &batch_type);
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
        assert_eq!(left.add_dimension(&right).unwrap().extent(), 10);
        left.require_less_than_or_equal(&DimensionValue::constant(8).unwrap()).unwrap();

        // Operation-aware capability methods preserve the operation's fresh result identity instead of constructing a
        // second eager-only result type.
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right_type =
            DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        let add = DimensionAddOperation::new(&left_type, &right_type).unwrap();
        let sum = left.add_dimension_with(&right, &add).unwrap();
        assert_eq!(sum.r#type(), &add.result_type());
        assert_eq!(sum.extent(), 10);

        // Concrete backend capability implementations retain the operation-owned diagnostics for invalid observed
        // extents admitted by otherwise valid operand bounds.
        let subtract = DimensionSubtractOperation::new(&left_type, &right_type).unwrap();
        let error = DimensionValue::new(left_type.clone(), 1)
            .unwrap()
            .subtract_dimension_with(&right, &subtract)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left >= right; observed left=1, right=3".to_string(),
            }),
        );
        let zero = DimensionValue::new(right_type.clone(), 0).unwrap();
        let floor_divide = DimensionFloorDivideOperation::new(&left_type, &right_type).unwrap();
        let error = left.floor_divide_dimension_with(&zero, &floor_divide).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
        let remainder = DimensionRemainderOperation::new(&left_type, &right_type).unwrap();
        let error = left.remainder_dimension_with(&zero, &remainder).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }
}
