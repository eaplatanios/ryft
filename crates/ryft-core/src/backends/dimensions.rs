//! Contains the host representation and homogeneous operation family for first-class runtime dimensions.
//!
//! [`DimensionValue`] is an ordinary scalar SSA value whose [`DimensionType`] defines one
//! [`DimensionVariable`]. Arithmetic produces fresh bounded variables through explicit program operations.

use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::{Operation, Parameter};

use crate::contexts::{Context, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::ConstantOperation;
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::Operation;
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Concretizable, Value};
use crate::tracing::TracingContext;
use crate::types::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};

// TODO(eaplatanios): Review this module.

/// Largest runtime dimension extent that every supported backend representation can carry.
///
/// Host values use [`usize`], while compiled dimension SSA uses a signed 64-bit scalar. On narrower hosts every
/// [`usize`] fits; on 64-bit hosts this excludes values that cannot be lowered without changing their meaning.
pub const MAX_DIMENSION_EXTENT: usize = if usize::BITS < i64::BITS { usize::MAX } else { i64::MAX as usize };

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

/// Canonical operation name for [`DimensionArithmetic::Add`].
pub const DIMENSION_ADD_OPERATION_NAME: &str = "dimension_add";

/// Canonical operation name for [`DimensionArithmetic::Subtract`].
pub const DIMENSION_SUBTRACT_OPERATION_NAME: &str = "dimension_subtract";

/// Canonical operation name for [`DimensionArithmetic::SubtractClamped`].
pub const DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME: &str = "dimension_subtract_clamped";

/// Canonical operation name for [`DimensionArithmetic::Multiply`].
pub const DIMENSION_MULTIPLY_OPERATION_NAME: &str = "dimension_multiply";

/// Canonical operation name for [`DimensionArithmetic::Power`].
pub const DIMENSION_POWER_OPERATION_NAME: &str = "dimension_power";

/// Canonical operation name for [`DimensionArithmetic::FloorDivide`].
pub const DIMENSION_FLOOR_DIVIDE_OPERATION_NAME: &str = "dimension_floor_divide";

/// Canonical operation name for [`DimensionArithmetic::Remainder`].
pub const DIMENSION_REMAINDER_OPERATION_NAME: &str = "dimension_remainder";

/// Canonical operation name for [`DimensionArithmetic::Minimum`].
pub const DIMENSION_MINIMUM_OPERATION_NAME: &str = "dimension_minimum";

/// Canonical operation name for [`DimensionArithmetic::Maximum`].
pub const DIMENSION_MAXIMUM_OPERATION_NAME: &str = "dimension_maximum";

// TODO(eaplatanios): Why do all this instead of leveraging distinct `Operation` types for each operation? This feels
//  a bit like an indirection though maybe I'm missing something.
/// Arithmetic function computed by a [`DimensionArithmeticOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum DimensionArithmetic {
    /// Checked addition.
    Add,

    /// Checked subtraction, which rejects a negative result.
    Subtract,

    /// Subtraction clamped at zero.
    SubtractClamped,

    /// Checked multiplication.
    Multiply,

    /// Checked exponentiation.
    Power,

    /// Floor division by a nonzero extent.
    FloorDivide,

    /// Remainder by a nonzero extent.
    Remainder,

    /// Minimum.
    Minimum,

    /// Maximum.
    Maximum,
}

impl DimensionArithmetic {
    /// Returns this arithmetic operation's canonical program name.
    fn operation_name(self) -> &'static str {
        match self {
            Self::Add => DIMENSION_ADD_OPERATION_NAME,
            Self::Subtract => DIMENSION_SUBTRACT_OPERATION_NAME,
            Self::SubtractClamped => DIMENSION_SUBTRACT_CLAMPED_OPERATION_NAME,
            Self::Multiply => DIMENSION_MULTIPLY_OPERATION_NAME,
            Self::Power => DIMENSION_POWER_OPERATION_NAME,
            Self::FloorDivide => DIMENSION_FLOOR_DIVIDE_OPERATION_NAME,
            Self::Remainder => DIMENSION_REMAINDER_OPERATION_NAME,
            Self::Minimum => DIMENSION_MINIMUM_OPERATION_NAME,
            Self::Maximum => DIMENSION_MAXIMUM_OPERATION_NAME,
        }
    }

    /// Returns a diagnostic name for the result variable.
    fn result_name(self, left: &DimensionType, right: &DimensionType) -> String {
        let left = left.variable();
        let right = right.variable();
        match self {
            Self::Add => format!("{left} + {right}"),
            Self::Subtract => format!("{left} - {right}"),
            Self::SubtractClamped => format!("max(0, {left} - {right})"),
            Self::Multiply => format!("{left} * {right}"),
            Self::Power => format!("{left} ^ {right}"),
            Self::FloorDivide => format!("{left} // {right}"),
            Self::Remainder => format!("{left} % {right}"),
            Self::Minimum => format!("min({left}, {right})"),
            Self::Maximum => format!("max({left}, {right})"),
        }
    }

    /// Returns the user-facing action used in checked arithmetic diagnostics.
    fn action(self) -> &'static str {
        match self {
            Self::Add => "adding runtime dimensions",
            Self::Subtract => "subtracting runtime dimensions",
            Self::SubtractClamped => "subtracting clamped runtime dimensions",
            Self::Multiply => "multiplying runtime dimensions",
            Self::Power => "raising a runtime dimension to a dimension power",
            Self::FloorDivide => "floor-dividing runtime dimensions",
            Self::Remainder => "taking the remainder of runtime dimensions",
            Self::Minimum => "taking the minimum of runtime dimensions",
            Self::Maximum => "taking the maximum of runtime dimensions",
        }
    }

    /// Derives sound bounds containing every successful backend-representable result.
    fn infer_bounds(self, left: &DimensionType, right: &DimensionType) -> Result<DimensionBounds, DimensionError> {
        let (left_lower, left_maximum) = representable_extent_range(left.bounds())?;
        let (right_lower, right_maximum) = representable_extent_range(right.bounds())?;
        let overflow = || DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while deriving '{}' result bounds with operands {left}, {right}",
                self.operation_name(),
            ),
        };
        let (lower, maximum) = match self {
            Self::Add => (
                left_lower.checked_add(right_lower).ok_or_else(overflow)?,
                left_maximum.saturating_add(right_maximum).min(MAX_DIMENSION_EXTENT),
            ),
            Self::Subtract => {
                if left_maximum < right_lower {
                    return Err(DimensionError::RequirementViolation {
                        message: format!(
                            "{} >= {} is impossible from declared bounds",
                            left.variable(),
                            right.variable()
                        ),
                    });
                }
                (left_lower.saturating_sub(right_maximum), left_maximum - right_lower)
            }
            Self::SubtractClamped => {
                (left_lower.saturating_sub(right_maximum), left_maximum.saturating_sub(right_lower))
            }
            Self::Multiply => (
                left_lower.checked_mul(right_lower).ok_or_else(overflow)?,
                left_maximum.saturating_mul(right_maximum).min(MAX_DIMENSION_EXTENT),
            ),
            Self::Power => {
                let lower = if right_maximum == 0 {
                    1
                } else if left_lower == 0 {
                    0
                } else if left_lower == 1 {
                    1
                } else {
                    checked_power(left_lower, right_lower).ok_or_else(overflow)?
                };
                let maximum = if right_maximum == 0 || left_maximum == 1 {
                    1
                } else if left_maximum == 0 {
                    usize::from(right_lower == 0)
                } else {
                    checked_power(left_maximum, right_maximum).unwrap_or(usize::MAX).min(MAX_DIMENSION_EXTENT)
                };
                (lower, maximum)
            }
            Self::FloorDivide => {
                let positive_right_lower = positive_divisor_lower_bound(right, right_maximum)?;
                (left_lower / right_maximum, left_maximum / positive_right_lower)
            }
            Self::Remainder => {
                positive_divisor_lower_bound(right, right_maximum)?;
                (0, left_maximum.min(right_maximum - 1))
            }
            Self::Minimum => (left_lower.min(right_lower), left_maximum.min(right_maximum)),
            Self::Maximum => (left_lower.max(right_lower), left_maximum.max(right_maximum)),
        };
        if lower > MAX_DIMENSION_EXTENT {
            return Err(overflow());
        }
        bounds_from_extrema(lower, maximum)
    }

    /// Evaluates this arithmetic operation using checked host arithmetic.
    fn evaluate(
        self,
        left_type: &DimensionType,
        left: usize,
        right_type: &DimensionType,
        right: usize,
    ) -> Result<usize, DimensionError> {
        let overflow = || DimensionError::ArithmeticOverflow {
            message: format!(
                "dimension arithmetic overflow while {} with operands {}={}, {}={}",
                self.action(),
                left_type.variable(),
                left,
                right_type.variable(),
                right,
            ),
        };
        match self {
            Self::Add => left.checked_add(right).ok_or_else(overflow),
            Self::Subtract => left.checked_sub(right).ok_or_else(|| {
                requirement_violation(
                    format!("{} >= {}", left_type.variable(), right_type.variable()),
                    left_type,
                    left,
                    right_type,
                    right,
                )
            }),
            Self::SubtractClamped => Ok(left.saturating_sub(right)),
            Self::Multiply => left.checked_mul(right).ok_or_else(overflow),
            Self::Power => checked_power(left, right).ok_or_else(overflow),
            Self::FloorDivide => {
                if right == 0 {
                    Err(requirement_violation(
                        format!("{} > 0", right_type.variable()),
                        left_type,
                        left,
                        right_type,
                        right,
                    ))
                } else {
                    Ok(left / right)
                }
            }
            Self::Remainder => {
                if right == 0 {
                    Err(requirement_violation(
                        format!("{} > 0", right_type.variable()),
                        left_type,
                        left,
                        right_type,
                        right,
                    ))
                } else {
                    Ok(left % right)
                }
            }
            Self::Minimum => Ok(left.min(right)),
            Self::Maximum => Ok(left.max(right)),
        }
    }
}

/// Returns the inclusive range of portable extents admitted by `bounds`.
fn representable_extent_range(bounds: DimensionBounds) -> Result<(usize, usize), DimensionError> {
    if bounds.lower() > MAX_DIMENSION_EXTENT {
        return Err(DimensionError::ExtentExceedsBackendWidth { value: bounds.lower(), maximum: MAX_DIMENSION_EXTENT });
    }
    let maximum = bounds.upper().map(|upper| upper - 1).unwrap_or(MAX_DIMENSION_EXTENT).min(MAX_DIMENSION_EXTENT);
    Ok((bounds.lower(), maximum))
}

/// Constructs bounds from inclusive extrema.
fn bounds_from_extrema(lower: usize, maximum: usize) -> Result<DimensionBounds, DimensionError> {
    DimensionBounds::new(lower, maximum.checked_add(1))
}

/// Computes `base.pow(exponent)` without narrowing `exponent`.
fn checked_power(mut base: usize, mut exponent: usize) -> Option<usize> {
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
fn positive_divisor_lower_bound(divisor: &DimensionType, maximum: usize) -> Result<usize, DimensionError> {
    if maximum == 0 {
        Err(DimensionError::RequirementViolation {
            message: format!("{} > 0 is impossible from declared bounds", divisor.variable()),
        })
    } else {
        Ok(divisor.bounds().lower().max(1))
    }
}

/// Constructs an observed requirement failure.
fn requirement_violation(
    requirement: String,
    left_type: &DimensionType,
    left: usize,
    right_type: &DimensionType,
    right: usize,
) -> DimensionError {
    DimensionError::RequirementViolation {
        message: format!("{requirement}; observed {}={left}, {}={right}", left_type.variable(), right_type.variable(),),
    }
}

/// Validates the types of a binary dimension operation's operands.
fn validate_binary_input_types(
    operation_name: &str,
    input_types: &[DimensionType],
    left: &DimensionType,
    right: &DimensionType,
) -> Result<(), TypeError> {
    check_count!("input", input_types, 2, TypeError);
    input_types.iter().zip([left, right]).enumerate().try_for_each(|(index, (actual, expected))| {
        if expected.is_refined_by(actual) {
            Ok(())
        } else {
            Err(TypeError::invalid(format!(
                "'{operation_name}' input {index} has type {actual} but the operation was constructed for type \
                 {expected}",
            )))
        }
    })
}

/// Operation that applies one checked binary [`DimensionArithmetic`] function to runtime dimensions.
///
/// Every arithmetic function has the same program contract: two dimension operands, one fresh dimension result,
/// bounds inferred from the operand types, checked host evaluation, and ordinary partial-evaluation behavior. Keeping
/// those shared semantics in one payload avoids parallel nominal operation types while [`Operation::name`] still
/// exposes a distinct primitive name for each function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionArithmeticOperation {
    /// Arithmetic function computed by this operation.
    arithmetic: DimensionArithmetic,

    /// Expected left operand type.
    left: DimensionType,

    /// Expected right operand type.
    right: DimensionType,

    /// Fresh variable defined by this operation's result.
    result: DimensionVariable,
}

impl DimensionArithmeticOperation {
    /// Creates an operation for one application and derives its fresh bounded result variable.
    pub fn new(
        arithmetic: DimensionArithmetic,
        left: &DimensionType,
        right: &DimensionType,
    ) -> Result<Self, DimensionError> {
        let result = DimensionVariable::new(arithmetic.result_name(left, right), arithmetic.infer_bounds(left, right)?);
        Ok(Self { arithmetic, left: left.clone(), right: right.clone(), result })
    }

    /// Returns the arithmetic function computed by this operation.
    #[inline]
    pub fn arithmetic(&self) -> DimensionArithmetic {
        self.arithmetic
    }

    /// Returns the expected left operand type.
    #[inline]
    pub fn left_type(&self) -> &DimensionType {
        &self.left
    }

    /// Returns the expected right operand type.
    #[inline]
    pub fn right_type(&self) -> &DimensionType {
        &self.right
    }

    /// Returns this operation's result type.
    #[inline]
    pub fn result_type(&self) -> DimensionType {
        DimensionType::new(self.result.clone())
    }
}

impl Display for DimensionArithmeticOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<DimensionType> for DimensionArithmeticOperation {
    #[inline]
    fn name(&self) -> &'static str {
        self.arithmetic.operation_name()
    }

    fn infer_output_types(
        &self,
        input_types: &[DimensionType],
        _region_interfaces: &[RegionInterface<DimensionType>],
    ) -> Result<Vec<DimensionType>, TypeError> {
        validate_binary_input_types(self.name(), input_types, &self.left, &self.right)?;
        Ok(vec![self.result_type()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        let result = self.result_type().rename_identities(renaming)?;
        Ok(Self {
            arithmetic: self.arithmetic,
            left: self.left.rename_identities(renaming)?,
            right: self.right.rename_identities(renaming)?,
            result: result.variable().clone(),
        })
    }
}

impl<O: Operation<DimensionType>> InterpretableOperation<EagerContext<DimensionValue, O>>
    for DimensionArithmeticOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<DimensionValue, O>>>(
        &self,
        _context: &EagerContext<DimensionValue, O>,
        _driver: &D,
        inputs: &[DimensionValue],
    ) -> Result<Vec<DimensionValue>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        self.infer_output_types(&[inputs[0].r#type().clone(), inputs[1].r#type().clone()], &[])?;
        let extent = self.arithmetic.evaluate(&self.left, inputs[0].extent(), &self.right, inputs[1].extent())?;
        Ok(vec![DimensionValue::new(self.result_type(), extent)?])
    }
}

impl<C: Context<Type = DimensionType, Operation: From<Self>>> PartiallyEvaluatableOperation<C>
    for DimensionArithmeticOperation
{
}

/// Homogeneous operation family for first-class runtime dimensions.
#[derive(Clone, Debug, Operation)]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    /// Dimension literal.
    Constant(ConstantOperation<V>),

    /// Checked binary dimension arithmetic.
    Arithmetic(DimensionArithmeticOperation),
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, StagingContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramBuilder;
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
    }

    #[test]
    fn test_dimension_arithmetic() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(2, Some(9)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(1, Some(5)).unwrap()));
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        let context = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();

        let add = DimensionArithmeticOperation::new(DimensionArithmetic::Add, &left_type, &right_type).unwrap();
        assert_eq!(add.result_type().bounds(), DimensionBounds::new(3, Some(13)).unwrap());
        assert_eq!(context.bind(add, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 10);

        let subtract =
            DimensionArithmeticOperation::new(DimensionArithmetic::Subtract, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(subtract, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 4,);

        let clamped =
            DimensionArithmeticOperation::new(DimensionArithmetic::SubtractClamped, &right_type, &left_type).unwrap();
        assert_eq!(context.bind(clamped, Vec::new(), &[right.clone(), left.clone()]).unwrap()[0].extent(), 0,);

        let multiply =
            DimensionArithmeticOperation::new(DimensionArithmetic::Multiply, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(multiply, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 21,);

        let power = DimensionArithmeticOperation::new(DimensionArithmetic::Power, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(power, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 343,);

        let floor_divide =
            DimensionArithmeticOperation::new(DimensionArithmetic::FloorDivide, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(floor_divide, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 2,);

        let remainder =
            DimensionArithmeticOperation::new(DimensionArithmetic::Remainder, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(remainder, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 1,);

        let minimum = DimensionArithmeticOperation::new(DimensionArithmetic::Minimum, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(minimum, Vec::new(), &[left.clone(), right.clone()]).unwrap()[0].extent(), 3,);

        let maximum = DimensionArithmeticOperation::new(DimensionArithmetic::Maximum, &left_type, &right_type).unwrap();
        assert_eq!(context.bind(maximum, Vec::new(), &[left, right]).unwrap()[0].extent(), 7);

        let base_type = DimensionType::new(DimensionVariable::new("base", DimensionBounds::new(0, Some(3)).unwrap()));
        let exponent_type =
            DimensionType::new(DimensionVariable::new("exponent", DimensionBounds::new(0, Some(3)).unwrap()));
        let power = DimensionArithmeticOperation::new(DimensionArithmetic::Power, &base_type, &exponent_type).unwrap();
        assert_eq!(power.result_type().bounds(), DimensionBounds::new(0, Some(5)).unwrap());
        assert_eq!(
            context
                .bind(
                    power,
                    Vec::new(),
                    &[DimensionValue::new(base_type, 0).unwrap(), DimensionValue::new(exponent_type, 0).unwrap(),],
                )
                .unwrap()[0]
                .extent(),
            1,
        );
    }

    #[test]
    fn test_dimension_arithmetic_errors() {
        let operand_type =
            DimensionType::new(DimensionVariable::new("operand", DimensionBounds::new(0, Some(5)).unwrap()));
        let other_type = DimensionType::new(DimensionVariable::new("other", DimensionBounds::new(0, Some(5)).unwrap()));
        let context = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();

        let subtract =
            DimensionArithmeticOperation::new(DimensionArithmetic::Subtract, &operand_type, &other_type).unwrap();
        let error = context
            .bind(
                subtract,
                Vec::new(),
                &[
                    DimensionValue::new(operand_type.clone(), 1).unwrap(),
                    DimensionValue::new(other_type.clone(), 3).unwrap(),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "operand >= other; observed operand=1, other=3".to_string(),
            }),
        );

        let add = DimensionArithmeticOperation::new(DimensionArithmetic::Add, &operand_type, &other_type).unwrap();
        let unexpected_type =
            DimensionType::new(DimensionVariable::new("unexpected", DimensionBounds::new(0, Some(6)).unwrap()));
        assert_eq!(
            add.infer_output_types(&[unexpected_type.clone(), other_type.clone()], &[]),
            Err(TypeError::invalid(format!(
                "'dimension_add' input 0 has type {unexpected_type} but the operation was constructed for type \
                 {operand_type}",
            ))),
        );

        let zero_type = DimensionType::new(DimensionVariable::new("zero", DimensionBounds::new(0, Some(1)).unwrap()));
        assert_eq!(
            DimensionArithmeticOperation::new(DimensionArithmetic::FloorDivide, &operand_type, &zero_type),
            Err(DimensionError::RequirementViolation {
                message: "zero > 0 is impossible from declared bounds".to_string(),
            }),
        );

        let maximum_type =
            DimensionType::new(DimensionVariable::new("maximum", DimensionBounds::at_least(MAX_DIMENSION_EXTENT)));
        let one_type = DimensionType::new(DimensionVariable::new("one", DimensionBounds::new(1, Some(2)).unwrap()));
        assert_eq!(
            DimensionArithmeticOperation::new(DimensionArithmetic::Add, &maximum_type, &one_type),
            Err(DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow while deriving 'dimension_add' result bounds with operands \
                     {maximum_type}, {one_type}",
                ),
            }),
        );
    }

    #[test]
    fn test_dimension_program_and_partial_evaluation() {
        let left_type = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(1, Some(9)).unwrap()));
        let right_type = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(1, Some(5)).unwrap()));
        let operation = DimensionArithmeticOperation::new(DimensionArithmetic::Add, &left_type, &right_type).unwrap();
        let result_type = operation.result_type();

        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = builder.add_input(left_type.clone());
        let right = builder.add_input(right_type.clone());
        let result = builder.add_instruction(operation, Vec::new(), vec![left, right]).unwrap()[0];
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                vec![result],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Dimension arithmetic is ordinary SSA: the result defines a fresh variable and runs through the same
        // interpretation and partial-evaluation paths as any other homogeneous program.
        assert_eq!(program.output_types(), vec![result_type.clone()]);
        assert_ne!(result_type.variable(), left_type.variable());
        assert_ne!(result_type.variable(), right_type.variable());
        let left = DimensionValue::new(left_type.clone(), 7).unwrap();
        let right = DimensionValue::new(right_type.clone(), 3).unwrap();
        assert_eq!(program.interpret(vec![left.clone(), right.clone()]).unwrap()[0].extent(), 10);
        assert_eq!(
            program
                .interpret(vec![DimensionValue::constant(7).unwrap(), DimensionValue::constant(3).unwrap()])
                .unwrap()[0]
                .extent(),
            10,
        );

        let evaluation = program.partially_evaluate(&[PartialValue::Known(left), PartialValue::Known(right)]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(
            evaluation.outputs(),
            &[PartialEvaluationOutput::Known(DimensionValue::new(result_type.clone(), 10).unwrap())],
        );

        let (traced_type, traced_program) = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::trace(
            |left| {
                let right = left.context().lift(DimensionValue::constant(2)?)?;
                let operation = DimensionArithmeticOperation::new(
                    DimensionArithmetic::Add,
                    &left.r#type().into_owned(),
                    &right.r#type().into_owned(),
                )?;
                let mut outputs = left.context().stage_operation(operation, Vec::new(), &[&left, &right])?;
                Ok(outputs.remove(0))
            },
            left_type.clone(),
        )
        .unwrap();
        assert_eq!(traced_type.bounds(), DimensionBounds::new(3, Some(11)).unwrap());
        assert_eq!(traced_program.interpret(DimensionValue::new(left_type, 6).unwrap()).unwrap().extent(), 8,);
    }
}
