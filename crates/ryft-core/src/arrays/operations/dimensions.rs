use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::DimensionOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};
use crate::arrays::types::ir::ArrayIrType;
use crate::backends::Array;
use crate::batching::{
    BatchableOperation, BatchingContext, BatchingDriver, BatchingError, ReplicatedDimensionBatchingPolicy,
};
use crate::contexts::{Context, ProjectedContext};
use crate::operations::dimensions::checked_power;
use crate::operations::{
    Add, DimensionAddOperation, DimensionDivFloorOperation, DimensionFromScalar, DimensionFromScalarOperation,
    DimensionMax, DimensionMaxOperation, DimensionMin, DimensionMinOperation, DimensionMulOperation, DimensionPow,
    DimensionPowOperation, DimensionRemOperation, DimensionRequirement, DimensionRequirementOperation,
    DimensionSaturatingSub, DimensionSaturatingSubOperation, DimensionSize, DimensionSizeOperation,
    DimensionSubOperation, DimensionToScalar, Div, Mul, Rem, Sub,
};
use crate::programs::{Operation, OperationProjection, ProgramError, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review from here onwards.

/// Composite batching executes homogeneous dimension operations only over replicated projected values. A mapped
/// dimension is rejected by [`ReplicatedDimensionBatchingPolicy`] before this rule is called because representing one
/// extent per batch item would require a ragged value model.
impl<C: Context<Type = ArrayIrType>>
    BatchableOperation<ProjectedContext<C, DimensionType>, ReplicatedDimensionBatchingPolicy>
    for DimensionOperation<DimensionValue>
where
    C::Constant: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Operation: OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
{
    fn batch<D: BatchingDriver<ProjectedContext<C, DimensionType>, ReplicatedDimensionBatchingPolicy>>(
        &self,
        context: &BatchingContext<ProjectedContext<C, DimensionType>, ReplicatedDimensionBatchingPolicy>,
        _driver: &D,
        inputs: &[<C::Value as ValueProjection<DimensionType>>::Projected],
    ) -> Result<Vec<<C::Value as ValueProjection<DimensionType>>::Projected>, BatchingError> {
        context.parent().bind(self.clone(), Vec::new(), inputs).map_err(Into::into)
    }
}

impl Add for DimensionValue {
    fn add(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionAddOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_add(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while adding dimensions with operands {}={}, {}={}",
                self.r#type().variable(),
                self.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })?;
        Ok(Self::new(result_type, extent)?)
    }
}

impl Sub for DimensionValue {
    fn sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSubOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_sub(right.extent()).ok_or_else(|| {
            let left_variable = self.r#type().variable().to_string();
            let right_variable = right.r#type().variable().to_string();
            DimensionError::RequirementViolation {
                message: format!(
                    "{left_variable} >= {right_variable}; observed {left_variable}={}, {right_variable}={}",
                    self.extent(),
                    right.extent(),
                ),
            }
        })?;
        Ok(Self::new(result_type, extent)?)
    }
}

impl DimensionSaturatingSub for DimensionValue {
    fn dimension_saturating_sub(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionSaturatingSubOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(result_type, self.extent().saturating_sub(right.extent()))?)
    }
}

impl Mul for DimensionValue {
    fn mul(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMulOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent = self.extent().checked_mul(right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while multiplying dimensions with operands {}={}, {}={}",
                self.r#type().variable(),
                self.extent(),
                right.r#type().variable(),
                right.extent(),
            ),
        })?;
        Ok(Self::new(result_type, extent)?)
    }
}

impl DimensionPow for DimensionValue {
    fn dimension_pow(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionPowOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        let extent =
            checked_power(self.extent(), right.extent()).ok_or_else(|| DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow while raising a dimension to a dimension power with operands \
                     {}={}, {}={}",
                    self.r#type().variable(),
                    self.extent(),
                    right.r#type().variable(),
                    right.extent(),
                ),
            })?;
        Ok(Self::new(result_type, extent)?)
    }
}

impl Div for DimensionValue {
    fn div(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionDivFloorOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        if right.extent() == 0 {
            let left_variable = self.r#type().variable().to_string();
            let right_variable = right.r#type().variable().to_string();
            return Err(DimensionError::RequirementViolation {
                message: format!(
                    "{right_variable} > 0; observed {left_variable}={}, {right_variable}={}",
                    self.extent(),
                    right.extent(),
                ),
            }
            .into());
        }
        Ok(Self::new(result_type, self.extent() / right.extent())?)
    }
}

impl Rem for DimensionValue {
    fn rem(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionRemOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        if right.extent() == 0 {
            let left_variable = self.r#type().variable().to_string();
            let right_variable = right.r#type().variable().to_string();
            return Err(DimensionError::RequirementViolation {
                message: format!(
                    "{right_variable} > 0; observed {left_variable}={}, {right_variable}={}",
                    self.extent(),
                    right.extent(),
                ),
            }
            .into());
        }
        Ok(Self::new(result_type, self.extent() % right.extent())?)
    }
}

impl DimensionMin for DimensionValue {
    fn dimension_min(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMinOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(result_type, self.extent().min(right.extent()))?)
    }
}

impl DimensionMax for DimensionValue {
    fn dimension_max(&self, right: &Self) -> Result<Self, ProgramError> {
        let operation = DimensionMaxOperation::new(&self.r#type(), &right.r#type())?;
        let inputs = &[self.r#type().clone(), right.r#type().clone()];
        let result_type = operation.infer_output_types(inputs, &[])?.remove(0);
        Ok(Self::new(result_type, self.extent().max(right.extent()))?)
    }
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

impl_dimension_operator!(Add, add, Add, add);
impl_dimension_operator!(Sub, sub, Sub, sub);
impl_dimension_operator!(Mul, mul, Mul, mul);
impl_dimension_operator!(Div, div, Div, div);
impl_dimension_operator!(Rem, rem, Rem, rem);

impl DimensionRequirement for DimensionValue {
    fn require_equal(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::equal(&self.r#type(), &right.r#type())
            .evaluate_extents(self.extent(), Some(right.extent()))
            .map_err(Into::into)
    }

    fn require_less_than_or_equal(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::less_than_or_equal(&self.r#type(), &right.r#type())
            .evaluate_extents(self.extent(), Some(right.extent()))
            .map_err(Into::into)
    }

    fn require_divisible_by(&self, right: &Self) -> Result<(), ProgramError> {
        DimensionRequirementOperation::divisible_by(&self.r#type(), &right.r#type())
            .evaluate_extents(self.extent(), Some(right.extent()))
            .map_err(Into::into)
    }

    fn require_bounds(&self, bounds: DimensionBounds) -> Result<(), ProgramError> {
        DimensionRequirementOperation::bounds(&self.r#type(), bounds)
            .evaluate_extents(self.extent(), None)
            .map_err(Into::into)
    }
}

impl<A: DimensionSize<usize> + Value<Type = ArrayType>> DimensionSize for ArrayIrValue<A> {
    fn dimension_size<AxisValue: Into<crate::Axis>>(&self, axis: AxisValue) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let input_type = array.r#type();
        let operation = DimensionSizeOperation::new(input_type.as_ref(), axis)?;
        let extent = <A as DimensionSize<usize>>::dimension_size(array, operation.axis())?;
        Ok(Self::Dimension(DimensionValue::new(operation.result_type().clone(), extent)?))
    }
}

impl DimensionToScalar<Array> for DimensionValue {
    fn to_scalar(&self) -> Result<Array, ProgramError> {
        // `DimensionValue::new` enforces the portable extent ceiling, which is no greater than `i64::MAX`.
        Ok(Array::scalar(i64::try_from(self.extent()).unwrap()))
    }
}

impl DimensionFromScalar<DimensionValue> for Array {
    fn to_dimension(&self, result: DimensionVariable) -> Result<DimensionValue, ProgramError> {
        let operation = DimensionFromScalarOperation::new(result);
        DimensionFromScalarOperation::validate_input_type(self.r#type().as_ref())?;
        let (scalar, extent) = match self.r#type().data_type() {
            DataType::I8 => {
                let value = self.elements::<i8>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            DataType::I16 => {
                let value = self.elements::<i16>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            DataType::I32 => {
                let value = self.elements::<i32>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            DataType::I64 => {
                let value = self.elements::<i64>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            DataType::U8 => {
                let value = self.elements::<u8>()?[0];
                (value.to_string(), Ok(usize::from(value)))
            }
            DataType::U16 => {
                let value = self.elements::<u16>()?[0];
                (value.to_string(), Ok(usize::from(value)))
            }
            DataType::U32 => {
                let value = self.elements::<u32>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            DataType::U64 => {
                let value = self.elements::<u64>()?[0];
                (value.to_string(), usize::try_from(value))
            }
            _ => unreachable!("dimension_from_scalar input type is validated before reading its payload"),
        };
        let extent = extent.map_err(|_| ProgramError::InvalidArgument {
            message: format!(
                "'{}' scalar input must be a nonnegative host-representable extent but is {scalar}",
                operation.name(),
            ),
        })?;
        Ok(DimensionValue::new(operation.result_type().clone(), extent)?)
    }
}

impl<A: Value<Type = ArrayType>> DimensionToScalar for ArrayIrValue<A>
where
    DimensionValue: DimensionToScalar<A>,
{
    fn to_scalar(&self) -> Result<Self, ProgramError> {
        let dimension = <Self as ValueProjection<DimensionType>>::projected(self)?;
        Ok(Self::Array(<DimensionValue as DimensionToScalar<A>>::to_scalar(dimension)?))
    }
}

impl<A: DimensionFromScalar<DimensionValue> + Value<Type = ArrayType>> DimensionFromScalar for ArrayIrValue<A> {
    fn to_dimension(&self, result: DimensionVariable) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Dimension(<A as DimensionFromScalar<DimensionValue>>::to_dimension(array, result)?))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::contexts::EagerContext;
    use crate::tracing::Trace;

    use super::*;

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
        assert!(matches!(program.instructions()[1].operation(), DimensionOperation::Mul(_)));
        assert!(matches!(program.instructions()[2].operation(), DimensionOperation::Sub(_)));
        assert!(matches!(program.instructions()[3].operation(), DimensionOperation::DivFloor(_)));
        assert!(matches!(program.instructions()[4].operation(), DimensionOperation::Rem(_)));
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
