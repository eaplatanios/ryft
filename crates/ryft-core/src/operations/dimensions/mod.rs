//! First-class dimension SSA operations.
//!
//! These operations compute and validate runtime extents used by shape-carrying array operations. They are ordinary
//! program operations over [`DimensionType`], not integer array operations and not a parallel symbolic-expression
//! language.

use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT};
use crate::macros::check_count;
use crate::parameters::Parameter;
use crate::programs::{Operation, Type, TypeError, TypeIdentityRenaming};

// TODO(eaplatanios): Review this module.

pub mod dimension_add;
pub mod dimension_div_floor;
pub mod dimension_from_scalar;
pub mod dimension_max;
pub mod dimension_min;
pub mod dimension_mul;
pub mod dimension_pow;
pub mod dimension_rem;
pub mod dimension_requirement;
pub mod dimension_saturating_sub;
pub mod dimension_size;
pub mod dimension_sub;
pub mod dimension_to_scalar;

pub use dimension_add::{DIMENSION_ADD_OPERATION_NAME, DimensionAddOperation};
pub use dimension_div_floor::{DIMENSION_DIV_FLOOR_OPERATION_NAME, DimensionDivFloorOperation};
pub use dimension_from_scalar::{
    DIMENSION_FROM_SCALAR_OPERATION_NAME, DimensionFromScalar, DimensionFromScalarOperation,
};
pub use dimension_max::{DIMENSION_MAX_OPERATION_NAME, DimensionMax, DimensionMaxOperation};
pub use dimension_min::{DIMENSION_MIN_OPERATION_NAME, DimensionMin, DimensionMinOperation};
pub use dimension_mul::{DIMENSION_MUL_OPERATION_NAME, DimensionMulOperation};
pub use dimension_pow::{DIMENSION_POW_OPERATION_NAME, DimensionPow, DimensionPowOperation};
pub use dimension_rem::{DIMENSION_REM_OPERATION_NAME, DimensionRemOperation};
pub use dimension_requirement::{
    DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME, DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
    DIMENSION_REQUIRE_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME, DimensionRequirement,
    DimensionRequirementOperation, DimensionRequirementPredicate,
};
pub use dimension_saturating_sub::{
    DIMENSION_SATURATING_SUB_OPERATION_NAME, DimensionSaturatingSub, DimensionSaturatingSubOperation,
};
pub use dimension_size::{DIMENSION_SIZE_OPERATION_NAME, DimensionSize, DimensionSizeOperation};
pub use dimension_sub::{DIMENSION_SUB_OPERATION_NAME, DimensionSubOperation};
pub use dimension_to_scalar::{
    DIMENSION_TO_SCALAR_OPERATION_NAME, DimensionToScalar, DimensionToScalarOperation, RUNTIME_DIMENSION_DATA_TYPE,
};

/// Shared contract implemented by binary first-class-dimension arithmetic operations.
///
/// Each nominal operation owns its bounds formula and is paired with a value capability. This trait centralizes the
/// common two-input type validation and fresh-result contract without imposing a concrete backend value
/// representation.
pub trait ArithmeticDimensionOperation: Operation<Type = DimensionType> {
    /// Returns the declared left operand type.
    fn left_type(&self) -> &DimensionType;

    /// Returns the declared right operand type.
    fn right_type(&self) -> &DimensionType;

    /// Returns the diagnostic name used for a freshly inferred result variable.
    fn result_name(&self) -> &str;

    /// Returns the bounds of a freshly inferred result variable.
    fn result_bounds(&self) -> DimensionBounds;

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
        Ok(vec![DimensionType::new(DimensionVariable::new(self.result_name(), self.result_bounds()))])
    }
}

/// Shared identity-bearing inference and effect metadata stored by every binary dimension arithmetic operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
pub(crate) struct ArithmeticDimensionOperationMetadata {
    /// Expected left operand type.
    left: DimensionType,

    /// Expected right operand type.
    right: DimensionType,

    /// Diagnostic name assigned to the result variable when output inference creates it.
    result_name: String,

    /// Authoritative bounds assigned to the result variable when output inference creates it.
    result_bounds: DimensionBounds,

    /// Whether the admitted operand bounds leave a checked runtime arithmetic failure possible.
    requires_runtime_assertion: bool,
}

impl ArithmeticDimensionOperationMetadata {
    /// Constructs shared arithmetic metadata used to infer one fresh result variable and classify its effects.
    pub(crate) fn new(
        left: &DimensionType,
        right: &DimensionType,
        result_name: String,
        result_bounds: DimensionBounds,
        requires_runtime_assertion: bool,
    ) -> Self {
        Self { left: left.clone(), right: right.clone(), result_name, result_bounds, requires_runtime_assertion }
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

    /// Returns the diagnostic name used for a freshly inferred result variable.
    #[inline]
    pub(crate) fn result_name(&self) -> &str {
        &self.result_name
    }

    /// Returns the bounds of a freshly inferred result variable.
    #[inline]
    pub(crate) fn result_bounds(&self) -> DimensionBounds {
        self.result_bounds
    }

    /// Returns whether the admitted operand bounds leave a checked runtime arithmetic failure possible.
    #[inline]
    pub(crate) fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Applies one simultaneous identity renaming to both operands.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(Self {
            left: self.left.rename_identities(renaming)?,
            right: self.right.rename_identities(renaming)?,
            result_name: self.result_name.clone(),
            result_bounds: self.result_bounds,
            requires_runtime_assertion: self.requires_runtime_assertion,
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

/// Returns the largest portable extent admitted by `r#type` when its upper bound is finite.
#[inline]
pub(crate) fn maximum_extent(r#type: &DimensionType) -> Option<usize> {
    r#type.bounds().upper()?.checked_sub(1).map(|maximum| maximum.min(MAX_DIMENSION_EXTENT))
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

#[cfg(test)]
fn test_dimension_type(name: &'static str, lower: usize, upper: usize) -> DimensionType {
    DimensionType::new(DimensionVariable::new(name, DimensionBounds::new(lower, Some(upper)).unwrap()))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::{Effect, Effects, Operation, TypeError, TypeIdentityRenaming};

    use super::*;

    #[test]
    fn test_arithmetic_dimension_operation() {
        let left = test_dimension_type("left", 2, 9);
        let right = test_dimension_type("right", 1, 5);
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        let result = Operation::infer_output_types(&operation, &[left.clone(), right.clone()], &[]).unwrap();
        assert_eq!(result[0].bounds(), operation.result_bounds());
        assert_ne!(result[0].variable(), left.variable());
        assert_ne!(result[0].variable(), right.variable());

        let unexpected = test_dimension_type("unexpected", 0, 6);
        assert_eq!(
            Operation::infer_output_types(&operation, &[unexpected.clone(), right.clone()], &[]),
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

    #[test]
    fn test_arithmetic_dimension_operation_effects() {
        let bounded_left = test_dimension_type("bounded_left", 2, 9);
        let bounded_right = test_dimension_type("bounded_right", 1, 5);
        let safe_subtrahend = test_dimension_type("safe_subtrahend", 1, 4);
        let safe_minuend = test_dimension_type("safe_minuend", 5, 9);
        let maybe_zero = test_dimension_type("maybe_zero", 0, 5);
        let unbounded = DimensionType::new(DimensionVariable::new("unbounded", DimensionBounds::unbounded()));
        let assertion = Effects::single(Effect::OrderedAssertion);

        // Arithmetic is pure exactly when operand bounds prove that its checked eager operation is total.
        assert_eq!(DimensionAddOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionAddOperation::new(&unbounded, &bounded_right).unwrap().effects(), assertion);
        assert_eq!(DimensionSubOperation::new(&safe_minuend, &safe_subtrahend).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionSubOperation::new(&bounded_left, &bounded_right).unwrap().effects(), assertion);
        assert_eq!(
            DimensionSaturatingSubOperation::new(&bounded_left, &bounded_right).unwrap().effects(),
            Effects::PURE,
        );
        assert_eq!(DimensionMulOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionMulOperation::new(&unbounded, &bounded_right).unwrap().effects(), assertion);
        assert_eq!(DimensionPowOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionPowOperation::new(&unbounded, &bounded_right).unwrap().effects(), assertion);
        assert_eq!(DimensionDivFloorOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionDivFloorOperation::new(&bounded_left, &maybe_zero).unwrap().effects(), assertion);
        assert_eq!(DimensionRemOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionRemOperation::new(&bounded_left, &maybe_zero).unwrap().effects(), assertion);
        assert_eq!(DimensionMinOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
        assert_eq!(DimensionMaxOperation::new(&bounded_left, &bounded_right).unwrap().effects(), Effects::PURE);
    }
}
