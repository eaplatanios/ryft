//! Contains the host representation and closed operation family for first-class runtime dimensions.
//!
//! [`DimensionValue`] is an ordinary scalar SSA value whose [`DimensionType`] defines one
//! [`DimensionVariable`]. Arithmetic produces fresh bounded variables through nominal operations in
//! [`crate::operations::dimensions`].

use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::{Operation, Parameter};

use crate::contexts::EagerContext;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::ConstantOperation;
use crate::operations::dimensions::{
    ArithmeticDimensionOperation, DimensionAddOperation, DimensionFloorDivideOperation, DimensionMaximumOperation,
    DimensionMinimumOperation, DimensionMultiplyOperation, DimensionPowerOperation, DimensionRemainderOperation,
    DimensionRequirementOperation, DimensionSubtractClampedOperation, DimensionSubtractOperation,
};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::Operation;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Concretizable, Value};
use crate::tracing::TracingContext;
use crate::types::{DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT};

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
    type DispatchDomain = EagerContext<Self, DimensionOperation<Self>>;
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

/// Implements eager host interpretation for one nominal binary dimension arithmetic primitive.
macro_rules! impl_arithmetic_dimension_interpretation {
    // This branch validates two concrete inputs and evaluates the statically selected primitive implementation.
    ($operation:ty) => {
        impl<O: Operation<DimensionType>> InterpretableOperation<EagerContext<DimensionValue, O>> for $operation {
            fn interpret<D: InterpretationDriver<EagerContext<DimensionValue, O>>>(
                &self,
                _context: &EagerContext<DimensionValue, O>,
                _driver: &D,
                inputs: &[DimensionValue],
            ) -> Result<Vec<DimensionValue>, ProgramError> {
                check_count!("input", inputs, 2, ProgramError);
                Operation::infer_output_types(self, &[inputs[0].r#type().clone(), inputs[1].r#type().clone()], &[])?;
                let extent = self.evaluate_extents(inputs[0].extent(), inputs[1].extent())?;
                Ok(vec![DimensionValue::new(self.result_type(), extent)?])
            }
        }
    };
}

impl_arithmetic_dimension_interpretation!(DimensionAddOperation);
impl_arithmetic_dimension_interpretation!(DimensionSubtractOperation);
impl_arithmetic_dimension_interpretation!(DimensionSubtractClampedOperation);
impl_arithmetic_dimension_interpretation!(DimensionMultiplyOperation);
impl_arithmetic_dimension_interpretation!(DimensionPowerOperation);
impl_arithmetic_dimension_interpretation!(DimensionFloorDivideOperation);
impl_arithmetic_dimension_interpretation!(DimensionRemainderOperation);
impl_arithmetic_dimension_interpretation!(DimensionMinimumOperation);
impl_arithmetic_dimension_interpretation!(DimensionMaximumOperation);

impl<O: Operation<DimensionType>> InterpretableOperation<EagerContext<DimensionValue, O>>
    for DimensionRequirementOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<DimensionValue, O>>>(
        &self,
        _context: &EagerContext<DimensionValue, O>,
        _driver: &D,
        inputs: &[DimensionValue],
    ) -> Result<Vec<DimensionValue>, ProgramError> {
        check_count!("input", inputs, self.input_count(), ProgramError);
        let input_types = inputs.iter().map(|input| input.r#type().clone()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        self.evaluate_extents(inputs[0].extent(), inputs.get(1).map(DimensionValue::extent))?;
        Ok(Vec::new())
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

    use crate::operations::dimensions::{DimensionAdd, DimensionRequirement};

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
    }
}
