//! Operations and value capabilities for [`DimensionType`] values. These operations compute array extents, infer their
//! [`DimensionBounds`], and validate runtime shape requirements. Dimension values are ordinary program inputs and
//! outputs, so shape computations can be traced and transformed.

use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};
use crate::macros::check_count;
use crate::parameters::Parameter;
use crate::programs::{Operation, Type, TypeError, TypeIdentityRenaming};

pub mod dimension_add;
pub mod dimension_div;
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
pub use dimension_div::{DIMENSION_DIV_OPERATION_NAME, DimensionDivOperation};
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

/// Shared contract implemented by binary arithmetic [`Operation`]s over [`DimensionValue`](crate::DimensionValue)s.
/// Each nominal operation owns its [`DimensionBounds`] formula and is paired with a value capability. This trait
/// centralizes the common two-input [`DimensionType`] validation and fresh-output contract without imposing a concrete
/// backend value representation.
pub trait ArithmeticDimensionOperation: Operation<Type = DimensionType> {
    /// Returns the declared left input [`DimensionType`] of this [`ArithmeticDimensionOperation`].
    fn left_type(&self) -> &DimensionType;

    /// Returns the declared right input [`DimensionType`] of this [`ArithmeticDimensionOperation`].
    fn right_type(&self) -> &DimensionType;

    /// Returns the diagnostic name used for a freshly inferred output variable for this
    /// [`ArithmeticDimensionOperation`].
    fn output_name(&self) -> &str;

    /// Returns the output [`DimensionBounds`] computed from the declared input [`DimensionType`]s when this
    /// [`ArithmeticDimensionOperation`] was constructed.
    fn output_bounds(&self) -> DimensionBounds;

    /// Infers output [`DimensionBounds`] from the actual input [`DimensionType`]s, which may refine the declared
    /// output [`DimensionBounds`] (i.e., the bounds returned by [`Self::output_bounds`]).
    fn infer_output_bounds(
        &self,
        left: &DimensionType,
        right: &DimensionType,
    ) -> Result<DimensionBounds, DimensionError>;

    /// Infers this operation's output [`DimensionType`]s after validating the provided input [`DimensionType`]s.
    fn infer_output_types(&self, input_types: &[DimensionType]) -> Result<Vec<DimensionType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        input_types.iter().zip([self.left_type(), self.right_type()]).enumerate().try_for_each(
            |(index, (actual, expected))| {
                if expected.is_refined_by(actual) {
                    Ok(())
                } else {
                    Err(TypeError::invalid(format!(
                        "`{}` input {index} has type {actual} but the operation was constructed for type {expected}",
                        self.name(),
                    )))
                }
            },
        )?;

        // Reusing an operation with narrower inputs must recompute its bounds formula. Its stored effect metadata
        // remains conservative as refinement never removes an assertion from the original operation.
        let bounds = self.infer_output_bounds(&input_types[0], &input_types[1])?;
        Ok(vec![DimensionType::new(self.output_name(), bounds)])
    }
}

/// Shared type-identity inference and effect metadata stored by every binary dimension arithmetic operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
pub(crate) struct ArithmeticDimensionOperationMetadata {
    /// Expected left input [`DimensionType`].
    left: DimensionType,

    /// Expected right input [`DimensionType`].
    right: DimensionType,

    /// Diagnostic name assigned to the output variable when output inference creates it.
    output_name: String,

    /// Output [`DimensionBounds`] computed at construction time. Type inference recomputes them when actual input
    /// bounds are narrower.
    output_bounds: DimensionBounds,

    /// Whether the admitted input bounds leave a checked runtime arithmetic failure possible.
    requires_runtime_assertion: bool,
}

impl ArithmeticDimensionOperationMetadata {
    /// Constructs a new [`ArithmeticDimensionOperationMetadata`] instance that is used to infer one fresh output
    /// variable and classify its effects.
    pub(crate) fn new(
        left: &DimensionType,
        right: &DimensionType,
        output_name: String,
        output_bounds: DimensionBounds,
        requires_runtime_assertion: bool,
    ) -> Self {
        Self { left: left.clone(), right: right.clone(), output_name, output_bounds, requires_runtime_assertion }
    }

    /// Returns the expected left input [`DimensionType`].
    pub(crate) fn left_type(&self) -> &DimensionType {
        &self.left
    }

    /// Returns the expected right input [`DimensionType`].
    pub(crate) fn right_type(&self) -> &DimensionType {
        &self.right
    }

    /// Returns the diagnostic name used for a freshly inferred output variable.
    pub(crate) fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Returns the output [`DimensionBounds`] computed from the declared input [`DimensionType`]s when this operation
    /// was constructed.
    pub(crate) fn output_bounds(&self) -> DimensionBounds {
        self.output_bounds
    }

    /// Returns whether the admitted input bounds leave a checked runtime arithmetic failure possible.
    pub(crate) fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Applies one simultaneous [`TypeIdentityRenaming`] to both input [`DimensionType`]s and returns a new
    /// [`ArithmeticDimensionOperationMetadata`] instance with the renamed types.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(Self {
            left: self.left.rename_identities(renaming)?,
            right: self.right.rename_identities(renaming)?,
            output_name: self.output_name.clone(),
            output_bounds: self.output_bounds,
            requires_runtime_assertion: self.requires_runtime_assertion,
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_arithmetic_dimension_operation_infer_output_types() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        let output =
            ArithmeticDimensionOperation::infer_output_types(&operation, &[left.clone(), right.clone()]).unwrap();
        assert_eq!(output[0].bounds(), operation.output_bounds());
        assert_ne!(output[0].variable(), left.variable());
        assert_ne!(output[0].variable(), right.variable());

        // Actual input types refine the output bounds, and every inference creates a fresh output identity.
        let inputs = [
            DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap()),
            DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap()),
        ];
        let output = ArithmeticDimensionOperation::infer_output_types(&operation, &inputs).unwrap();
        let repeated = ArithmeticDimensionOperation::infer_output_types(&operation, &inputs).unwrap();
        assert_eq!(output[0].extent(), Some(8));
        assert_ne!(output[0].variable(), repeated[0].variable());

        assert_eq!(
            ArithmeticDimensionOperation::infer_output_types(&operation, std::slice::from_ref(&left)),
            Err(TypeError::invalid("expected 2 inputs but got 1")),
        );
        let unexpected = DimensionType::new("unexpected", DimensionBounds::new(0, Some(6)).unwrap());
        assert_eq!(
            ArithmeticDimensionOperation::infer_output_types(&operation, &[unexpected.clone(), right.clone()]),
            Err(TypeError::invalid(format!(
                "`dimension_add` input 0 has type {unexpected} but the operation was constructed for type {left}",
            ))),
        );
        assert_eq!(
            ArithmeticDimensionOperation::infer_output_types(&operation, &[left, unexpected.clone()]),
            Err(TypeError::invalid(format!(
                "`dimension_add` input 1 has type {unexpected} but the operation was constructed for type {right}",
            ))),
        );
    }

    #[test]
    fn test_arithmetic_dimension_operation_metadata_rename_type_identities() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let metadata = ArithmeticDimensionOperationMetadata::new(
            &left,
            &right,
            "sum".to_string(),
            DimensionBounds::new(3, Some(13)).unwrap(),
            true,
        );
        let renamed_left = DimensionType::new("renamed_left", left.bounds());
        let renamed_right = DimensionType::new("renamed_right", right.bounds());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left.variable().clone(), renamed_left.variable().clone()).unwrap();
        renaming.insert(right.variable().clone(), renamed_right.variable().clone()).unwrap();
        let renamed = metadata.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.left_type(), &renamed_left);
        assert_eq!(renamed.right_type(), &renamed_right);
        assert_eq!(renamed.output_name(), metadata.output_name());
        assert_eq!(renamed.output_bounds(), metadata.output_bounds());
        assert_eq!(renamed.requires_runtime_assertion(), metadata.requires_runtime_assertion());
    }
}
