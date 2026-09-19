//! Array-universe implementations of the dimension operation family contracts.
//!
//! First-class dimensions project through the mixed array IR and batch only as replicated values. This module owns
//! standard operator wrappers, the array-to-dimension gateway, and concrete reference-array dimension-size queries.
//! Checked host-integer capability implementations live beside their dimension operation definitions.

// TODO(eaplatanios): Review this module.

use crate::arrays::arrays::Array;
use crate::arrays::batching::ReplicatedDimensionBatchingPolicy;
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::{ArrayIrOperation, DimensionOperation};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{DimensionType, DimensionVariable};
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::Axis;
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, ProjectedContext};
use crate::operations::{
    Add, DIMENSION_SIZE_OPERATION_NAME, DimensionAddOperation, DimensionArithmetic, DimensionDivOperation,
    DimensionFromScalar, DimensionFromScalarOperation, DimensionMax, DimensionMaxOperation, DimensionMin,
    DimensionMinOperation, DimensionMulOperation, DimensionPow, DimensionPowOperation, DimensionRemOperation,
    DimensionRequirementOperation, DimensionSaturatingSub, DimensionSaturatingSubOperation, DimensionSize,
    DimensionSizeOperation, DimensionSubOperation, DimensionToScalar, Div, Mul, Rem, Sub,
};
use crate::programs::{Operation, OperationProjection, ProgramError, TypeError, Typed, Value, ValueProjection};

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
    ) -> Result<BatchedOutputs<ProjectedContext<C, DimensionType>, ReplicatedDimensionBatchingPolicy>, BatchingError>
    {
        Ok(context.parent().bind(self.clone(), Vec::new(), inputs)?.into())
    }
}

/// Lifts one homogeneous first-class dimension operation directly into [`ArrayIrOperation`].
macro_rules! impl_dimension_operation_lift {
    // Each accepted item is a concrete homogeneous dimension operation type owned by `DimensionOperation`.
    ($($operation:ident),+ $(,)?) => {
        $(
            impl<A: Value<Type = ArrayType>> From<$operation> for ArrayIrOperation<A> {
                #[inline]
                fn from(operation: $operation) -> Self {
                    Self::Dimension(operation.into())
                }
            }
        )+
    };
}

// Every first-class dimension operation lifts directly into the composite family, so that generic composite code can
// state a plain `From<DimensionMulOperation>`-style bound without naming this family's dimension member. Each lift
// routes through the member family that owns the operation's payload, semantics, and rendering. `ConstantOperation`
// has its own lift next to the other constant constructors.
impl_dimension_operation_lift!(
    DimensionAddOperation,
    DimensionSubOperation,
    DimensionSaturatingSubOperation,
    DimensionMulOperation,
    DimensionPowOperation,
    DimensionDivOperation,
    DimensionRemOperation,
    DimensionMinOperation,
    DimensionMaxOperation,
    DimensionRequirementOperation,
);

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

/// Implements the composite dimension arithmetic capability for eager composite values, which have no context to
/// stage into and therefore project both operands, apply the member capability, and inject the result back.
macro_rules! impl_composite_dimension_arithmetic {
    // Each accepted item pairs one composite capability method with the member capability method that performs it.
    ($($method:ident => $member_method:ident),+ $(,)?) => {
        impl<A: Value<Type = ArrayType>> DimensionArithmetic for ArrayIrValue<A> {
            $(
                fn $method(&self, right: &Self) -> Result<Self, ProgramError> {
                    let left = <Self as ValueProjection<DimensionType>>::projected(self)?;
                    let right = <Self as ValueProjection<DimensionType>>::projected(right)?;
                    Ok(Self::Dimension(left.$member_method(right)?))
                }
            )+
        }
    };
}

impl_composite_dimension_arithmetic!(
    dimension_add => add,
    dimension_sub => sub,
    dimension_saturating_sub => dimension_saturating_sub,
    dimension_mul => mul,
    dimension_pow => dimension_pow,
    dimension_div => div,
    dimension_rem => rem,
    dimension_min => dimension_min,
    dimension_max => dimension_max,
);

impl<A: DimensionSize<usize> + Value<Type = ArrayType>> DimensionSize for ArrayIrValue<A> {
    fn dimension_size<AxisValue: Into<crate::Axis>>(&self, axis: AxisValue) -> Result<Self, ProgramError> {
        let array = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let input_type = array.r#type();
        let operation = DimensionSizeOperation::new(input_type.as_ref(), axis)?;
        let extent = <A as DimensionSize<usize>>::dimension_size(array, operation.axis())?;
        Ok(Self::Dimension(DimensionValue::new(operation.result_type().clone(), extent)?))
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
                "`{}` scalar input must be a nonnegative host-representable extent but is {scalar}",
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

impl DimensionSize<usize> for Array {
    fn dimension_size<AxisValue: Into<Axis>>(&self, axis: AxisValue) -> Result<usize, ProgramError> {
        let axis = axis.into();
        let position = axis.normalize(self.r#type().rank()).map_err(|_| {
            TypeError::invalid(format!(
                "`{DIMENSION_SIZE_OPERATION_NAME}` axis {axis} is out of bounds for rank {}",
                self.r#type().rank(),
            ))
        })?;
        let r#type = self.r#type();
        let dimension = &r#type.shape().dimensions()[position];
        dimension.value().ok_or_else(|| {
            TypeError::invalid(format!("materialized reference array has a dynamic dimension at axis {position}",))
                .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::types::dimensions::{DimensionBounds, DimensionError};
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
        let left_type = DimensionType::new("left", DimensionBounds::new(3, Some(9)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(1, Some(4)).unwrap());
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
        assert!(matches!(program.instructions()[3].operation(), DimensionOperation::Div(_)));
        assert!(matches!(program.instructions()[4].operation(), DimensionOperation::Rem(_)));
    }

    #[test]
    fn test_dimension_tracer_operator_propagates_construction_error() {
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(2)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(3, Some(5)).unwrap());
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
        let left_type = DimensionType::new("left", DimensionBounds::new(0, Some(10)).unwrap());
        let right_type = DimensionType::new("right", DimensionBounds::new(0, Some(10)).unwrap());
        let left = DimensionValue::new(left_type, 1).unwrap();
        let right = DimensionValue::new(right_type, 3).unwrap();

        let _ = left - right;
    }
}
