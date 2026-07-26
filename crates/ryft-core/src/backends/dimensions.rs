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

/// [`TracingContext`] over [`DimensionValue`]s and [`DimensionOperation`]s.
pub type DimensionTracingContext = TracingContext<DimensionValue, DimensionOperation<DimensionValue>>;

/// Checked host representation of a first-class runtime [`Dimension`](crate::Dimension) value. Its eager domain
/// performs checked host integer arithmetic without allocating an array or dispatching to a device backend. Fallible
/// capabilities such as [`DimensionAdd::add`] form the canonical arithmetic API. [`std::ops::Add`], [`std::ops::Sub`],
/// [`std::ops::Mul`], [`std::ops::Div`], and [`std::ops::Rem`] implementations provide panicking operator sugar for
/// both owned and borrowed values.
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

    /// Returns the [`DimensionType`] of this [`DimensionValue`].
    #[inline]
    pub fn r#type(&self) -> &DimensionType {
        &self.r#type
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

// TODO(eaplatanios): Review from here onwards.

/// Implements one concrete checked-arithmetic capability for [`DimensionValue`].
///
/// The evaluator receives both complete operand values so it can use their extents and type metadata without exposing
/// macro-internal local names in the invocation syntax.
macro_rules! impl_arithmetic_dimension_capability {
    ($capability:ident, $method:ident, $operation:ty, |$left:ident, $right:ident| $evaluate:expr $(,)?) => {
        impl $capability for DimensionValue {
            fn $method(&self, right: &Self) -> Result<Self, ProgramError> {
                let operation = <$operation>::new(&self.r#type, &right.r#type)?;
                let inputs = &[self.r#type.clone(), right.r#type.clone()];
                let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
                let $left: &DimensionValue = self;
                let $right: &DimensionValue = right;
                let extent: Result<usize, DimensionError> = $evaluate;
                Ok(Self::new(result_type, extent?)?)
            }
        }
    };
}

impl_arithmetic_dimension_capability! {
    DimensionAdd,
    add,
    DimensionAddOperation,
    |left, right| {
        left.extent().checked_add(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while adding runtime dimensions with operands {}={}, {}={}",
                left.r#type().variable(),
                left.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })
    },
}

impl_arithmetic_dimension_capability! {
    DimensionSubtract,
    sub,
    DimensionSubtractOperation,
    |left, right| {
        left.extent().checked_sub(right.extent()).ok_or_else(|| {
            requirement_violation(
                format!("{} >= {}", left.r#type().variable(), right.r#type().variable()),
                left.r#type(),
                left.extent(),
                right.r#type(),
                right.extent(),
            )
        })
    },
}

impl_arithmetic_dimension_capability! {
    DimensionSubtractClamped,
    sub_clamped,
    DimensionSubtractClampedOperation,
    |left, right| { Ok(left.extent().saturating_sub(right.extent())) },
}

impl_arithmetic_dimension_capability! {
    DimensionMultiply,
    mul,
    DimensionMultiplyOperation,
    |left, right| {
        left.extent().checked_mul(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while multiplying runtime dimensions with operands {}={}, {}={}",
                left.r#type().variable(),
                left.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })
    },
}

impl_arithmetic_dimension_capability! {
    DimensionPower,
    pow,
    DimensionPowerOperation,
    |left, right| {
        checked_power(left.extent(), right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while raising a runtime dimension to a dimension power with operands \
                 {}={}, {}={}",
                left.r#type().variable(),
                left.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })
    },
}

impl_arithmetic_dimension_capability! {
    DimensionFloorDivide,
    floor_divide,
    DimensionFloorDivideOperation,
    |left, right| {
        if right.extent() == 0 {
            Err(requirement_violation(
                format!("{} > 0", right.r#type().variable()),
                left.r#type(),
                left.extent(),
                right.r#type(),
                right.extent(),
            ))
        } else {
            Ok(left.extent() / right.extent())
        }
    },
}

impl_arithmetic_dimension_capability! {
    DimensionRemainder,
    remainder,
    DimensionRemainderOperation,
    |left, right| {
        if right.extent() == 0 {
            Err(requirement_violation(
                format!("{} > 0", right.r#type().variable()),
                left.r#type(),
                left.extent(),
                right.r#type(),
                right.extent(),
            ))
        } else {
            Ok(left.extent() % right.extent())
        }
    },
}

impl_arithmetic_dimension_capability! {
    DimensionMinimum,
    minimum,
    DimensionMinimumOperation,
    |left, right| { Ok(left.extent().min(right.extent())) },
}

impl_arithmetic_dimension_capability! {
    DimensionMaximum,
    maximum,
    DimensionMaximumOperation,
    |left, right| { Ok(left.extent().max(right.extent())) },
}

/// Implements one panicking standard operator as sugar for a fallible [`DimensionValue`] capability.
macro_rules! impl_dimension_operator {
    // This branch supports every owned/borrowed operand combination by delegating to one borrowed capability method.
    ($operator:ident, $operator_method:ident, $capability:ident, $capability_method:ident) => {
        impl std::ops::$operator for DimensionValue {
            type Output = DimensionValue;

            #[inline]
            fn $operator_method(self, right: DimensionValue) -> Self::Output {
                $capability::$capability_method(&self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl std::ops::$operator<&DimensionValue> for DimensionValue {
            type Output = DimensionValue;

            #[inline]
            fn $operator_method(self, right: &DimensionValue) -> Self::Output {
                $capability::$capability_method(&self, right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl std::ops::$operator<DimensionValue> for &DimensionValue {
            type Output = DimensionValue;

            #[inline]
            fn $operator_method(self, right: DimensionValue) -> Self::Output {
                $capability::$capability_method(self, &right).unwrap_or_else(|error| panic!("{error}"))
            }
        }

        impl std::ops::$operator<&DimensionValue> for &DimensionValue {
            type Output = DimensionValue;

            #[inline]
            fn $operator_method(self, right: &DimensionValue) -> Self::Output {
                $capability::$capability_method(self, right).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    };
}

impl_dimension_operator!(Add, add, DimensionAdd, add);
impl_dimension_operator!(Sub, sub, DimensionSubtract, sub);
impl_dimension_operator!(Mul, mul, DimensionMultiply, mul);
impl_dimension_operator!(Div, div, DimensionFloorDivide, floor_divide);
impl_dimension_operator!(Rem, rem, DimensionRemainder, remainder);

impl DimensionRequirement for DimensionValue {
    fn require_equal(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::equal(&self.r#type, &right.r#type)
            .evaluate_extents(self.extent, Some(right.extent))
            .map_err(Into::into)
    }

    fn require_less_than_or_equal(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::less_than_or_equal(&self.r#type, &right.r#type)
            .evaluate_extents(self.extent, Some(right.extent))
            .map_err(Into::into)
    }

    fn require_divisible_by(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::divisible_by(&self.r#type, &right.r#type)
            .evaluate_extents(self.extent, Some(right.extent))
            .map_err(Into::into)
    }

    fn require_bounds(&self, bounds: DimensionBounds) -> Result<(), ProgramError> {
        DimensionRequirementOperation::bounds(&self.r#type, bounds)
            .evaluate_extents(self.extent, None)
            .map_err(Into::into)
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
        DimensionAdd, DimensionFloorDivide, DimensionRemainder, DimensionRequirement, DimensionSubtract,
    };
    use crate::tracing::Trace;

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
        let error = left.floor_divide(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
        let error = left.remainder(&zero).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "right > 0; observed left=7, right=0".to_string() }),
        );
    }

    #[test]
    fn test_dimension_value_operators() {
        let left = DimensionValue::constant(7).unwrap();
        let right = DimensionValue::constant(3).unwrap();

        // Addition covers every owned/borrowed combination generated by the shared operator implementation.
        assert_eq!((left.clone() + right.clone()).extent(), 10);
        assert_eq!((left.clone() + &right).extent(), 10);
        assert_eq!((&left + right.clone()).extent(), 10);
        assert_eq!((&left + &right).extent(), 10);

        // The remaining standard operators preserve the checked dimension semantics of their fallible capabilities.
        assert_eq!((&left - &right).extent(), 4);
        assert_eq!((&left * &right).extent(), 21);
        assert_eq!((&left / &right).extent(), 2);
        assert_eq!((&left % &right).extent(), 1);
    }

    #[test]
    fn test_dimension_tracer_operators() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(3, Some(9)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(1, Some(4)).unwrap()));
        let (output_type, program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| {
                let sum = left.clone() + right.clone();
                let product = sum * right.clone();
                let difference = product - left.clone();
                let quotient = difference / right.clone();
                Ok(quotient % left)
            },
            (left_type.clone(), right_type.clone()),
        )
        .unwrap();

        assert_eq!(output_type.bounds(), DimensionBounds::new(0, Some(8)).unwrap());
        assert_eq!(
            program
                .interpret((DimensionValue::new(left_type, 7).unwrap(), DimensionValue::new(right_type, 3).unwrap(),))
                .unwrap()
                .extent(),
            0,
        );
        assert!(matches!(program.instructions()[0].operation(), DimensionOperation::Add(_)));
        assert!(matches!(program.instructions()[1].operation(), DimensionOperation::Multiply(_)));
        assert!(matches!(program.instructions()[2].operation(), DimensionOperation::Subtract(_)));
        assert!(matches!(program.instructions()[3].operation(), DimensionOperation::FloorDivide(_)));
        assert!(matches!(program.instructions()[4].operation(), DimensionOperation::Remainder(_)));
    }

    #[test]
    fn test_dimension_tracer_operator_propagates_construction_error() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(2)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(3, Some(5)).unwrap()));
        let result = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |(left, right)| Ok(left - right),
            (left_type, right_type),
        );

        let Err(error) = result else {
            panic!("expected impossible subtraction bounds to fail tracing");
        };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left >= right is impossible from declared bounds".to_string(),
            }),
        );
    }

    #[test]
    #[should_panic(expected = "left >= right; observed left=1, right=3")]
    fn test_dimension_value_operator_panics_on_capability_error() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right_type =
            DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));
        let left = DimensionValue::new(left_type, 1).unwrap();
        let right = DimensionValue::new(right_type, 3).unwrap();

        let _ = left - right;
    }
}
