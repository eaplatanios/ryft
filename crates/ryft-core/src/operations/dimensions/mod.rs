//! First-class dimension SSA operations.
//!
//! These operations compute and validate runtime extents used by shape-carrying array operations. They are ordinary
//! program operations over [`DimensionType`], not integer array operations and not a parallel symbolic-expression
//! language.

use crate::macros::check_count;
use crate::parameters::Parameter;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, TypeError};
use crate::types::{DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT};

pub mod add;
pub mod floor_divide;
pub mod maximum;
pub mod minimum;
pub mod multiply;
pub mod power;
pub mod remainder;
pub mod requirement;
pub mod subtract;
pub mod subtract_clamped;

pub use add::{DIMENSION_ADD_OPERATION_NAME, DimensionAdd, DimensionAddOperation};
pub use floor_divide::{DIMENSION_FLOOR_DIVIDE_OPERATION_NAME, DimensionFloorDivide, DimensionFloorDivideOperation};
pub use maximum::{DIMENSION_MAXIMUM_OPERATION_NAME, DimensionMaximum, DimensionMaximumOperation};
pub use minimum::{DIMENSION_MINIMUM_OPERATION_NAME, DimensionMinimum, DimensionMinimumOperation};
pub use multiply::{DIMENSION_MULTIPLY_OPERATION_NAME, DimensionMultiply, DimensionMultiplyOperation};
pub use power::{DIMENSION_POWER_OPERATION_NAME, DimensionPower, DimensionPowerOperation};
pub use remainder::{DIMENSION_REMAINDER_OPERATION_NAME, DimensionRemainder, DimensionRemainderOperation};
pub use requirement::{
    DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME, DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
    DIMENSION_REQUIRE_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME, DimensionRequirement,
    DimensionRequirementOperation, DimensionRequirementPredicate,
};
pub use subtract::{DIMENSION_SUBTRACT_OPERATION_NAME, DimensionSubtract, DimensionSubtractOperation};
pub use subtract_clamped::{
    DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME, DimensionSubtractClamped, DimensionSubtractClampedOperation,
};

/// Shared contract implemented by binary first-class-dimension arithmetic operations.
///
/// Each nominal operation owns its bounds formula and is paired with a value capability. This trait centralizes the
/// common two-input type validation and fresh-result contract without imposing a concrete backend value
/// representation.
pub trait ArithmeticDimensionOperation: Operation<DimensionType> {
    /// Returns the declared left operand type.
    fn left_type(&self) -> &DimensionType;

    /// Returns the declared right operand type.
    fn right_type(&self) -> &DimensionType;

    /// Returns the fresh result type defined by this operation.
    fn result_type(&self) -> DimensionType;

    /// Infers this operation's one fresh dimension result after validating both operand types.
    fn infer_output_types(&self, input_types: &[DimensionType]) -> Result<Vec<DimensionType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types.iter().zip([self.left_type(), self.right_type()]).enumerate().try_for_each(
            |(index, (actual, expected))| {
                if expected.is_refined_by(actual) {
                    Ok(())
                } else {
                    Err(TypeError::invalid(format!(
                        "'{}' input {index} has type {actual} but the operation was constructed for type {expected}",
                        self.name(),
                    )))
                }
            },
        )?;
        Ok(vec![self.result_type()])
    }
}

/// Shared identity-bearing metadata stored by every binary dimension arithmetic operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
pub(crate) struct ArithmeticDimensionOperationMetadata {
    /// Expected left operand type.
    left: DimensionType,

    /// Expected right operand type.
    right: DimensionType,

    /// Fresh variable defined by the operation's result.
    result: DimensionVariable,
}

impl ArithmeticDimensionOperationMetadata {
    /// Constructs shared arithmetic metadata with one fresh result variable.
    pub(crate) fn new(
        left: &DimensionType,
        right: &DimensionType,
        result_name: String,
        result_bounds: DimensionBounds,
    ) -> Self {
        Self { left: left.clone(), right: right.clone(), result: DimensionVariable::new(result_name, result_bounds) }
    }

    /// Returns the expected left operand type.
    #[inline]
    pub(crate) fn left_type(&self) -> &DimensionType {
        &self.left
    }

    /// Returns the expected right operand type.
    #[inline]
    pub(crate) fn right_type(&self) -> &DimensionType {
        &self.right
    }

    /// Returns the fresh result type.
    #[inline]
    pub(crate) fn result_type(&self) -> DimensionType {
        DimensionType::new(self.result.clone())
    }

    /// Applies one simultaneous identity renaming to both operands and the result.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        let result = self.result_type().rename_identities(renaming)?;
        Ok(Self {
            left: self.left.rename_identities(renaming)?,
            right: self.right.rename_identities(renaming)?,
            result: result.variable().clone(),
        })
    }
}

/// Returns the inclusive range of portable extents admitted by `bounds`.
pub(crate) fn representable_extent_range(bounds: DimensionBounds) -> Result<(usize, usize), DimensionError> {
    if bounds.lower() > MAX_DIMENSION_EXTENT {
        return Err(DimensionError::ExtentExceedsBackendWidth { value: bounds.lower(), maximum: MAX_DIMENSION_EXTENT });
    }
    let maximum = bounds.upper().map(|upper| upper - 1).unwrap_or(MAX_DIMENSION_EXTENT).min(MAX_DIMENSION_EXTENT);
    Ok((bounds.lower(), maximum))
}

/// Computes `base.pow(exponent)` without narrowing `exponent`.
pub(crate) fn checked_power(mut base: usize, mut exponent: usize) -> Option<usize> {
    let mut result = 1usize;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.checked_mul(base)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.checked_mul(base)?;
        }
    }
    Some(result)
}

/// Returns the smallest positive divisor admitted by `divisor`, rejecting an exact-zero divisor.
pub(crate) fn positive_divisor_lower_bound(divisor: &DimensionType, maximum: usize) -> Result<usize, DimensionError> {
    if maximum == 0 {
        Err(DimensionError::RequirementViolation {
            message: format!("{} > 0 is impossible from declared bounds", divisor.variable()),
        })
    } else {
        Ok(divisor.bounds().lower().max(1))
    }
}

/// Constructs a bounds-inference overflow diagnostic.
pub(crate) fn bounds_overflow(operation_name: &str, left: &DimensionType, right: &DimensionType) -> DimensionError {
    DimensionError::ArithmeticOverflow {
        message: format!(
            "dimension arithmetic overflow while deriving '{operation_name}' result bounds with operands {left}, \
             {right}",
        ),
    }
}

/// Constructs an observed binary requirement failure.
pub(crate) fn requirement_violation(
    requirement: String,
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> DimensionError {
    DimensionError::RequirementViolation {
        message: format!("{requirement}; observed {}={left}, {}={right}", left_type.variable(), right_type.variable()),
    }
}

#[cfg(test)]
fn test_dimension_type(name: &'static str, lower: usize, upper: usize) -> DimensionType {
    DimensionType::new(DimensionVariable::new(name, DimensionBounds::new(lower, Some(upper)).unwrap()))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::identities::TypeIdentityRenaming;
    use crate::programs::operations::Operation;
    use crate::programs::types::TypeError;

    use super::*;

    #[test]
    fn test_arithmetic_dimension_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        assert_eq!(
            Operation::infer_output_types(&operation, &[left.clone(), right.clone()], &[]),
            Ok(vec![operation.result_type()]),
        );

        let unexpected = test_dimension_type("unexpected", 0, 6);
        assert_eq!(
            Operation::infer_output_types(&operation, &[unexpected.clone(), right.clone()], &[],),
            Err(TypeError::invalid(format!(
                "'dimension_add' input 0 has type {unexpected} but the operation was constructed for type {left}",
            ))),
        );

        let renamed_left = test_dimension_type("renamed_left", 2, 9);
        let renamed_right = test_dimension_type("renamed_right", 1, 5);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left.variable().clone(), renamed_left.variable().clone()).unwrap();
        renaming.insert(right.variable().clone(), renamed_right.variable().clone()).unwrap();
        let renamed = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.left_type(), &renamed_left);
        assert_eq!(renamed.right_type(), &renamed_right);
    }
}
