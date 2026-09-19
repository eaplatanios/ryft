//! First-class dimension SSA operations.
//!
//! These operations compute and validate runtime extents used by shape-carrying array operations. They are ordinary
//! program operations over [`DimensionType`], not integer array operations and not a parallel symbolic-expression
//! language.

use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};
use crate::macros::check_count;
use crate::parameters::Parameter;
use crate::programs::{Operation, Type, TypeError, TypeIdentityRenaming};

// TODO(eaplatanios): Review this module.

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

    /// Returns the diagnostic name used for a freshly inferred output variable.
    fn output_name(&self) -> &str;

    /// Returns the output bounds computed from the declared input types when this operation was constructed.
    fn output_bounds(&self) -> DimensionBounds;

    /// Computes output bounds from the actual input types, which may refine the declared input bounds.
    fn infer_output_bounds(
        &self,
        left: &DimensionType,
        right: &DimensionType,
    ) -> Result<DimensionBounds, DimensionError>;

    /// Infers this operation's one fresh dimension result after validating both operand types.
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
        // remains conservative: refinement never removes an assertion from the original operation.
        let bounds = self.infer_output_bounds(&input_types[0], &input_types[1])?;
        Ok(vec![DimensionType::new(self.output_name(), bounds)])
    }
}

/// Shared type-identity inference and effect metadata stored by every binary dimension arithmetic operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
pub(crate) struct ArithmeticDimensionOperationMetadata {
    /// Expected left operand type.
    left: DimensionType,

    /// Expected right operand type.
    right: DimensionType,

    /// Diagnostic name assigned to the output variable when output inference creates it.
    output_name: String,

    /// Output bounds computed at construction; inference recomputes them when actual input bounds are narrower.
    output_bounds: DimensionBounds,

    /// Whether the admitted operand bounds leave a checked runtime arithmetic failure possible.
    requires_runtime_assertion: bool,
}

impl ArithmeticDimensionOperationMetadata {
    /// Constructs shared arithmetic metadata used to infer one fresh result variable and classify its effects.
    pub(crate) fn new(
        left: &DimensionType,
        right: &DimensionType,
        output_name: String,
        output_bounds: DimensionBounds,
        requires_runtime_assertion: bool,
    ) -> Self {
        Self { left: left.clone(), right: right.clone(), output_name, output_bounds, requires_runtime_assertion }
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

    /// Returns the diagnostic name used for a freshly inferred output variable.
    #[inline]
    pub(crate) fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Returns the output bounds computed from the declared input types when this operation was constructed.
    #[inline]
    pub(crate) fn output_bounds(&self) -> DimensionBounds {
        self.output_bounds
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
            output_name: self.output_name.clone(),
            output_bounds: self.output_bounds,
            requires_runtime_assertion: self.requires_runtime_assertion,
        })
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, DimensionValue};
    use crate::contexts::EagerContext;
    use crate::operations::math::add::Add;
    use crate::operations::math::div::Div;
    use crate::operations::math::mul::Mul;
    use crate::operations::math::rem::Rem;
    use crate::operations::math::sub::Sub;
    use crate::programs::{EffectClass, EffectClasses, Operation, TypeError, TypeIdentityRenaming, ValueProjection};
    use crate::tracing::Trace;

    use super::*;

    #[test]
    fn test_arithmetic_dimension_operation() {
        let left = DimensionType::new("left", DimensionBounds::new(2, Some(9)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let operation = DimensionAddOperation::new(&left, &right).unwrap();
        let result = Operation::infer_output_types(&operation, &[left.clone(), right.clone()], &[]).unwrap();
        assert_eq!(result[0].bounds(), operation.output_bounds());
        assert_ne!(result[0].variable(), left.variable());
        assert_ne!(result[0].variable(), right.variable());

        // All arithmetic formulas use the actual input bounds when a retained operation is specialized. Results
        // remain fresh definitions, and conservative assertion effects are preserved on the retained operation.
        let declared_left = DimensionType::new("left", DimensionBounds::new(1, Some(9)).unwrap());
        let declared_right = DimensionType::new("right", DimensionBounds::new(1, Some(5)).unwrap());
        let exact_left = DimensionType::new("left", DimensionBounds::new(6, Some(7)).unwrap());
        let exact_right = DimensionType::new("right", DimensionBounds::new(2, Some(3)).unwrap());
        macro_rules! check_refinement {
            // Check each concrete arithmetic operation with the same declarations and exact input refinements.
            ($operation:ident, $extent:literal) => {{
                let operation = $operation::new(&declared_left, &declared_right).unwrap();
                let inputs = [exact_left.clone(), exact_right.clone()];
                let result = Operation::infer_output_types(&operation, &inputs, &[]).unwrap();
                let repeated = Operation::infer_output_types(&operation, &inputs, &[]).unwrap();
                assert_eq!(result[0].extent(), Some($extent));
                assert_ne!(result[0].variable(), repeated[0].variable());
            }};
        }
        check_refinement!(DimensionAddOperation, 8);
        check_refinement!(DimensionSubOperation, 4);
        check_refinement!(DimensionSaturatingSubOperation, 4);
        check_refinement!(DimensionMulOperation, 12);
        check_refinement!(DimensionPowOperation, 36);
        check_refinement!(DimensionDivOperation, 3);
        check_refinement!(DimensionRemOperation, 0);
        check_refinement!(DimensionMinOperation, 2);
        check_refinement!(DimensionMaxOperation, 6);

        let checked_subtraction = DimensionSubOperation::new(&declared_left, &declared_right).unwrap();
        let result = Operation::infer_output_types(&checked_subtraction, &[exact_left, exact_right], &[]).unwrap();
        assert_eq!(result[0].extent(), Some(4));
        assert_eq!(checked_subtraction.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion),);

        let unexpected = DimensionType::new("unexpected", DimensionBounds::new(0, Some(6)).unwrap());
        assert_eq!(
            Operation::infer_output_types(&operation, &[unexpected.clone(), right.clone()], &[]),
            Err(TypeError::invalid(format!(
                "`dimension_add` input 0 has type {unexpected} but the operation was constructed for type {left}",
            ))),
        );

        let renamed_left = DimensionType::new("renamed_left", DimensionBounds::new(2, Some(9)).unwrap());
        let renamed_right = DimensionType::new("renamed_right", DimensionBounds::new(1, Some(5)).unwrap());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(left.variable().clone(), renamed_left.variable().clone()).unwrap();
        renaming.insert(right.variable().clone(), renamed_right.variable().clone()).unwrap();
        let renamed = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.left_type(), &renamed_left);
        assert_eq!(renamed.right_type(), &renamed_right);
    }

    #[test]
    fn test_arithmetic_dimension_operation_effects() {
        let bounded_left = DimensionType::new("bounded_left", DimensionBounds::new(2, Some(9)).unwrap());
        let bounded_right = DimensionType::new("bounded_right", DimensionBounds::new(1, Some(5)).unwrap());
        let safe_subtrahend = DimensionType::new("safe_subtrahend", DimensionBounds::new(1, Some(4)).unwrap());
        let safe_minuend = DimensionType::new("safe_minuend", DimensionBounds::new(5, Some(9)).unwrap());
        let maybe_zero = DimensionType::new("maybe_zero", DimensionBounds::new(0, Some(5)).unwrap());
        let unbounded = DimensionType::new("unbounded", DimensionBounds::unbounded());
        let assertion = EffectClasses::single(EffectClass::OrderedAssertion);

        // Arithmetic is pure exactly when operand bounds prove that its checked eager operation is total.
        assert_eq!(
            DimensionAddOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionAddOperation::new(&unbounded, &bounded_right).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionSubOperation::new(&safe_minuend, &safe_subtrahend).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionSubOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionSaturatingSubOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE,
        );
        assert_eq!(
            DimensionMulOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionMulOperation::new(&unbounded, &bounded_right).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionPowOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionPowOperation::new(&unbounded, &bounded_right).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionDivOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionDivOperation::new(&bounded_left, &maybe_zero).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionRemOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(DimensionRemOperation::new(&bounded_left, &maybe_zero).unwrap().effects().classes(), assertion);
        assert_eq!(
            DimensionMinOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
        assert_eq!(
            DimensionMaxOperation::new(&bounded_left, &bounded_right).unwrap().effects().classes(),
            EffectClasses::NONE
        );
    }

    #[test]
    fn test_composite_dimension_arithmetic() {
        let left = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(6).unwrap());
        let right = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(4).unwrap());
        let two = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        let left = ValueProjection::<DimensionType>::into_projected(left).unwrap();
        let right = ValueProjection::<DimensionType>::into_projected(right).unwrap();
        let two = ValueProjection::<DimensionType>::into_projected(two).unwrap();
        assert_eq!(left.add(&right).unwrap().extent(), 10);
        assert_eq!(left.sub(&right).unwrap().extent(), 2);
        assert_eq!(right.dimension_saturating_sub(&left).unwrap().extent(), 0);
        assert_eq!(left.mul(&right).unwrap().extent(), 24);
        assert_eq!(left.dimension_pow(&two).unwrap().extent(), 36);
        assert_eq!(left.div(&right).unwrap().extent(), 1);
        assert_eq!(left.rem(&right).unwrap().extent(), 2);
        assert_eq!(left.dimension_min(&right).unwrap().extent(), 4);
        assert_eq!(left.dimension_max(&right).unwrap().extent(), 6);
        let result = <ArrayIrValue<Array> as ValueProjection<DimensionType>>::from_projected(left.add(&right).unwrap());
        assert!(matches!(result, ArrayIrValue::Dimension(value) if value.extent() == 10));

        // An array member is rejected before dimension arithmetic can be invoked.
        assert!(matches!(
            ValueProjection::<DimensionType>::into_projected(ArrayIrValue::Array(Array::scalar(2.0_f64).unwrap())),
            Err(TypeError::Invalid { message }) if message == "expected dimension type but got array type",
        ));
    }

    #[test]
    fn test_composite_dimension_arithmetic_stages_ordinary_dimension_operands() {
        let rows = DimensionType::new("rows", DimensionBounds::new(5, Some(9)).unwrap());
        let columns = DimensionType::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |(rows, columns)| {
                let rows = ValueProjection::<DimensionType>::into_projected(rows)?;
                let columns = ValueProjection::<DimensionType>::into_projected(columns)?;
                let padded = rows.mul(&columns)?.add(&columns)?;
                let trimmed = padded.dimension_saturating_sub(&rows)?;
                Ok((
                    trimmed.div(&columns)?.into_value(),
                    trimmed.rem(&columns)?.into_value(),
                    rows.dimension_pow(&columns)?.into_value(),
                    rows.dimension_max(&columns)?.sub(&rows.dimension_min(&columns)?)?.into_value(),
                ))
            },
            (ArrayIrType::Dimension(rows), ArrayIrType::Dimension(columns)),
        )
        .unwrap();
        assert_eq!(output_type.0.to_string(), "dimension<max(0, rows * columns + columns - rows) / columns ∈ [0, 32)>",);
        let expected = indoc! {"
            lambda %0:dimension<rows ∈ [5, 9)>, %1:dimension<columns ∈ [1, 5)> .
            let %2:dimension<rows * columns ∈ [5, 33)> = dimension_mul %0 %1
                %3:dimension<rows * columns + columns ∈ [6, 37)> = dimension_add %2 %1
                %4:dimension<max(0, rows * columns + columns - rows) ∈ [0, 32)> = dimension_saturating_sub %3 %0
                %5:dimension<max(0, rows * columns + columns - rows) / columns ∈ [0, 32)> = dimension_div %4 %1
                %6:dimension<max(0, rows * columns + columns - rows) % columns ∈ [0, 4)> = dimension_rem %4 %1
                %7:dimension<rows ^ columns ∈ [5, 4097)> = dimension_pow %0 %1
                %8:dimension<max(rows, columns) ∈ [5, 9)> = dimension_max %0 %1
                %9:dimension<min(rows, columns) ∈ [1, 5)> = dimension_min %0 %1
                %10:dimension<max(rows, columns) - min(rows, columns) ∈ [1, 8)> = dimension_sub %8 %9
            in (%5, %6, %7, %10)
        "};
        assert_eq!(program.to_string(), expected.trim_end());
    }
}
