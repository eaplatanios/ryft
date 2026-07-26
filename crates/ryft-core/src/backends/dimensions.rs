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
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
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

/// Canonical operation name for [`DimensionRequirement::Equal`].
pub const DIMENSION_REQUIRE_EQUAL_OPERATION_NAME: &str = "dimension_require_equal";

/// Canonical operation name for [`DimensionRequirement::LessThanOrEqual`].
pub const DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME: &str = "dimension_require_less_than_or_equal";

/// Canonical operation name for [`DimensionRequirement::DivisibleBy`].
pub const DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME: &str = "dimension_require_divisible_by";

/// Canonical operation name for [`DimensionRequirement::Bounds`].
pub const DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME: &str = "dimension_require_bounds";

/// Arithmetic function computed by a [`DimensionArithmeticOperation`].
///
/// These functions deliberately share one operation payload because they have the same signature, identity behavior,
/// eager representation, and transformation rules. Distinct nominal payload types would duplicate those contracts and
/// add one outer-operation variant per function without making an invalid state unrepresentable. The tag selects only
/// the function-specific name, bounds transfer, and checked host calculation.
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

/// Outcome of proving a dimension requirement from the facts available at one program point.
enum DimensionRequirementProof {
    /// The requirement holds for every concrete value admitted by the available facts.
    Proven,

    /// The requirement fails for every concrete value admitted by the available facts.
    Disproven(DimensionError),

    /// The available facts admit both passing and failing concrete values.
    Inconclusive,
}

/// Exact-or-interval fact for one dimension SSA value.
struct AbstractDimensionValue {
    /// Declared operand type supplying the stable diagnostic name and same-variable relationship.
    r#type: DimensionType,

    /// Narrowest bounds known for this occurrence.
    bounds: DimensionBounds,

    /// Concrete extent when known from a literal or singleton interval.
    exact: Option<usize>,
}

impl AbstractDimensionValue {
    /// Constructs an abstract value from type-level bounds.
    fn from_type(r#type: &DimensionType) -> Self {
        let bounds = r#type.bounds();
        let exact =
            bounds.upper().filter(|upper| bounds.lower().checked_add(1) == Some(*upper)).map(|_| bounds.lower());
        Self { r#type: r#type.clone(), bounds, exact }
    }

    /// Constructs an abstract value from a partially evaluated input, preferring a resolved literal when available.
    fn from_partial<C: Context<Type = DimensionType, Constant = DimensionValue>>(
        context: &PartialEvaluationContext<C>,
        declared_type: &DimensionType,
        value: &PartialEvaluationValue<C::Value>,
    ) -> Self {
        let actual_type = value.r#type();
        let mut abstract_value = Self { r#type: declared_type.clone(), bounds: actual_type.bounds(), exact: None };
        abstract_value.exact = abstract_value
            .bounds
            .upper()
            .filter(|upper| abstract_value.bounds.lower().checked_add(1) == Some(*upper))
            .map(|_| abstract_value.bounds.lower());
        if let Some(value) = value.as_known()
            && let Some(value) = context.parent().resolve(value).into_constant()
        {
            abstract_value.exact = Some(value.extent());
        }
        abstract_value
    }

    /// Returns the inclusive minimum admitted by this value's facts.
    #[inline]
    fn minimum(&self) -> usize {
        self.exact.unwrap_or_else(|| self.bounds.lower())
    }

    /// Returns the inclusive maximum admitted by this value's interval.
    #[inline]
    fn maximum(&self) -> usize {
        self.exact.unwrap_or_else(|| self.bounds.upper().map_or(usize::MAX, |upper| upper - 1))
    }
}

/// Predicate enforced by a [`DimensionRequirementOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum DimensionRequirement {
    /// Requires `left == right`.
    Equal,

    /// Requires `left <= right`.
    LessThanOrEqual,

    /// Requires a positive `divisor` that divides `dividend` exactly.
    DivisibleBy,

    /// Requires `input` to lie within `bounds`.
    Bounds(DimensionBounds),
}

/// Zero-result requirement over first-class runtime dimensions.
///
/// Every [`DimensionRequirement`] shares one operation payload because the predicates have the same program contract:
/// they consume dimensions, produce no values, use the same three-way proof lattice, report the same ordered assertion
/// effect when inconclusive, and follow the same eager and partial-evaluation rules. The predicate tag selects only the
/// checked relation. Private operands plus the public constructors ensure that binary predicates always have a right
/// operand and bounds predicates never do, without multiplying nominal operation types and dispatch variants.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionRequirementOperation {
    /// Predicate enforced by this operation.
    requirement: DimensionRequirement,

    /// Expected left or sole input operand type.
    left: DimensionType,

    /// Expected right operand type for binary predicates.
    right: Option<DimensionType>,
}

impl DimensionRequirementOperation {
    /// Constructs an equality requirement.
    #[inline]
    pub fn equal(left: &DimensionType, right: &DimensionType) -> Self {
        Self { requirement: DimensionRequirement::Equal, left: left.clone(), right: Some(right.clone()) }
    }

    /// Constructs a less-than-or-equal requirement.
    #[inline]
    pub fn less_than_or_equal(left: &DimensionType, right: &DimensionType) -> Self {
        Self { requirement: DimensionRequirement::LessThanOrEqual, left: left.clone(), right: Some(right.clone()) }
    }

    /// Constructs a positive-divisibility requirement.
    #[inline]
    pub fn divisible_by(dividend: &DimensionType, divisor: &DimensionType) -> Self {
        Self { requirement: DimensionRequirement::DivisibleBy, left: dividend.clone(), right: Some(divisor.clone()) }
    }

    /// Constructs an explicit-bounds requirement.
    #[inline]
    pub fn bounds(input: &DimensionType, bounds: DimensionBounds) -> Self {
        Self { requirement: DimensionRequirement::Bounds(bounds), left: input.clone(), right: None }
    }

    /// Returns the predicate enforced by this operation.
    #[inline]
    pub fn requirement(&self) -> DimensionRequirement {
        self.requirement
    }

    /// Returns the expected left or sole input operand type.
    #[inline]
    pub fn left_type(&self) -> &DimensionType {
        &self.left
    }

    /// Returns the expected right operand type for a binary predicate.
    #[inline]
    pub fn right_type(&self) -> Option<&DimensionType> {
        self.right.as_ref()
    }

    /// Returns this requirement's canonical program name.
    fn operation_name(&self) -> &'static str {
        match self.requirement {
            DimensionRequirement::Equal => DIMENSION_REQUIRE_EQUAL_OPERATION_NAME,
            DimensionRequirement::LessThanOrEqual => DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME,
            DimensionRequirement::DivisibleBy => DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
            DimensionRequirement::Bounds(_) => DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME,
        }
    }

    /// Proves, disproves, or retains this requirement from type-level facts.
    fn prove_from_types(&self) -> DimensionRequirementProof {
        let left = AbstractDimensionValue::from_type(&self.left);
        match self.requirement {
            DimensionRequirement::Equal => {
                Self::prove_equal(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirement::LessThanOrEqual => {
                Self::prove_less_than_or_equal(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirement::DivisibleBy => {
                Self::prove_divisible_by(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirement::Bounds(bounds) => Self::prove_bounds(&left, bounds),
        }
    }

    /// Proves, disproves, or retains this requirement from partial-evaluation facts.
    fn prove_from_partial<C: Context<Type = DimensionType, Constant = DimensionValue>>(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<DimensionRequirementProof, ProgramError> {
        check_count!("input", inputs, self.input_count(), ProgramError);
        let left = AbstractDimensionValue::from_partial(context, &self.left, &inputs[0]);
        Ok(match self.requirement {
            DimensionRequirement::Equal => Self::prove_equal(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirement::LessThanOrEqual => Self::prove_less_than_or_equal(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirement::DivisibleBy => Self::prove_divisible_by(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirement::Bounds(bounds) => Self::prove_bounds(&left, bounds),
        })
    }

    /// Returns this requirement's operand count.
    #[inline]
    fn input_count(&self) -> usize {
        1 + usize::from(self.right.is_some())
    }

    /// Validates this requirement's declared operand types.
    fn validate_input_types(&self, input_types: &[DimensionType]) -> Result<(), TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        std::iter::once(&self.left).chain(self.right.iter()).zip(input_types).enumerate().try_for_each(
            |(index, (expected, actual))| {
                if expected.is_refined_by(actual) {
                    Ok(())
                } else {
                    Err(TypeError::invalid(format!(
                        "'{}' input {index} has type {actual} but the operation was constructed for type {expected}",
                        self.operation_name(),
                    )))
                }
            },
        )
    }

    /// Evaluates this requirement against concrete operands.
    fn evaluate(&self, inputs: &[DimensionValue]) -> Result<(), DimensionError> {
        match self.requirement {
            DimensionRequirement::Equal => {
                Self::evaluate_equal(&self.left, inputs[0].extent(), self.right.as_ref().unwrap(), inputs[1].extent())
            }
            DimensionRequirement::LessThanOrEqual => Self::evaluate_less_than_or_equal(
                &self.left,
                inputs[0].extent(),
                self.right.as_ref().unwrap(),
                inputs[1].extent(),
            ),
            DimensionRequirement::DivisibleBy => Self::evaluate_divisible_by(
                &self.left,
                inputs[0].extent(),
                self.right.as_ref().unwrap(),
                inputs[1].extent(),
            ),
            DimensionRequirement::Bounds(bounds) => {
                let value = inputs[0].extent();
                if bounds.contains(value) {
                    Ok(())
                } else {
                    Err(DimensionError::BindingOutOfBounds {
                        variable: self.left.variable().to_string(),
                        value,
                        bounds,
                    })
                }
            }
        }
    }

    /// Proves an equality requirement.
    fn prove_equal(left: &AbstractDimensionValue, right: &AbstractDimensionValue) -> DimensionRequirementProof {
        if let (Some(left_extent), Some(right_extent)) = (left.exact, right.exact) {
            return match Self::evaluate_equal(&left.r#type, left_extent, &right.r#type, right_extent) {
                Ok(()) => DimensionRequirementProof::Proven,
                Err(error) => DimensionRequirementProof::Disproven(error),
            };
        }
        if left.r#type == right.r#type {
            DimensionRequirementProof::Proven
        } else if left.maximum() < right.minimum() || right.maximum() < left.minimum() {
            DimensionRequirementProof::Disproven(Self::static_violation(format!(
                "{} == {}",
                left.r#type.variable(),
                right.r#type.variable(),
            )))
        } else {
            DimensionRequirementProof::Inconclusive
        }
    }

    /// Proves a less-than-or-equal requirement.
    fn prove_less_than_or_equal(
        left: &AbstractDimensionValue,
        right: &AbstractDimensionValue,
    ) -> DimensionRequirementProof {
        if let (Some(left_extent), Some(right_extent)) = (left.exact, right.exact) {
            return match Self::evaluate_less_than_or_equal(&left.r#type, left_extent, &right.r#type, right_extent) {
                Ok(()) => DimensionRequirementProof::Proven,
                Err(error) => DimensionRequirementProof::Disproven(error),
            };
        }
        if left.maximum() <= right.minimum() {
            DimensionRequirementProof::Proven
        } else if left.minimum() > right.maximum() {
            DimensionRequirementProof::Disproven(Self::static_violation(format!(
                "{} <= {}",
                left.r#type.variable(),
                right.r#type.variable(),
            )))
        } else {
            DimensionRequirementProof::Inconclusive
        }
    }

    /// Proves a positive-divisibility requirement.
    fn prove_divisible_by(
        dividend: &AbstractDimensionValue,
        divisor: &AbstractDimensionValue,
    ) -> DimensionRequirementProof {
        if let (Some(dividend_extent), Some(divisor_extent)) = (dividend.exact, divisor.exact) {
            return match Self::evaluate_divisible_by(&dividend.r#type, dividend_extent, &divisor.r#type, divisor_extent)
            {
                Ok(()) => DimensionRequirementProof::Proven,
                Err(error) => DimensionRequirementProof::Disproven(error),
            };
        }
        if divisor.exact == Some(0) {
            DimensionRequirementProof::Disproven(Self::static_violation(format!(
                "{} > 0 for divisibility",
                divisor.r#type.variable(),
            )))
        } else if divisor.exact == Some(1)
            || (dividend.exact == Some(0) && divisor.minimum() > 0)
            || (dividend.r#type == divisor.r#type && dividend.minimum() > 0)
        {
            DimensionRequirementProof::Proven
        } else {
            DimensionRequirementProof::Inconclusive
        }
    }

    /// Proves an explicit-bounds requirement.
    fn prove_bounds(input: &AbstractDimensionValue, bounds: DimensionBounds) -> DimensionRequirementProof {
        if let Some(value) = input.exact {
            return if bounds.contains(value) {
                DimensionRequirementProof::Proven
            } else {
                DimensionRequirementProof::Disproven(DimensionError::BindingOutOfBounds {
                    variable: input.r#type.variable().to_string(),
                    value,
                    bounds,
                })
            };
        }
        if bounds.contains_bounds(input.bounds) {
            DimensionRequirementProof::Proven
        } else if input.maximum() < bounds.lower() || bounds.upper().is_some_and(|upper| input.minimum() >= upper) {
            DimensionRequirementProof::Disproven(Self::static_violation(format!(
                "{} in {bounds}",
                input.r#type.variable(),
            )))
        } else {
            DimensionRequirementProof::Inconclusive
        }
    }

    /// Evaluates an equality requirement.
    fn evaluate_equal(
        left_type: &DimensionType,
        left: usize,
        right_type: &DimensionType,
        right: usize,
    ) -> Result<(), DimensionError> {
        if left == right {
            Ok(())
        } else {
            Err(requirement_violation(
                format!("{} == {}", left_type.variable(), right_type.variable()),
                left_type,
                left,
                right_type,
                right,
            ))
        }
    }

    /// Evaluates a less-than-or-equal requirement.
    fn evaluate_less_than_or_equal(
        left_type: &DimensionType,
        left: usize,
        right_type: &DimensionType,
        right: usize,
    ) -> Result<(), DimensionError> {
        if left <= right {
            Ok(())
        } else {
            Err(requirement_violation(
                format!("{} <= {}", left_type.variable(), right_type.variable()),
                left_type,
                left,
                right_type,
                right,
            ))
        }
    }

    /// Evaluates a positive-divisibility requirement.
    fn evaluate_divisible_by(
        dividend_type: &DimensionType,
        dividend: usize,
        divisor_type: &DimensionType,
        divisor: usize,
    ) -> Result<(), DimensionError> {
        if divisor == 0 {
            Err(requirement_violation(
                format!("{} > 0 for divisibility", divisor_type.variable()),
                dividend_type,
                dividend,
                divisor_type,
                divisor,
            ))
        } else if dividend.is_multiple_of(divisor) {
            Ok(())
        } else {
            Err(requirement_violation(
                format!("{} % {} == 0", dividend_type.variable(), divisor_type.variable()),
                dividend_type,
                dividend,
                divisor_type,
                divisor,
            ))
        }
    }

    /// Constructs a requirement failure proven solely from declared bounds.
    #[inline]
    fn static_violation(requirement: String) -> DimensionError {
        DimensionError::RequirementViolation { message: format!("{requirement} is impossible from declared bounds") }
    }
}

impl Display for DimensionRequirementOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<DimensionType> for DimensionRequirementOperation {
    #[inline]
    fn name(&self) -> &'static str {
        self.operation_name()
    }

    fn infer_output_types(
        &self,
        input_types: &[DimensionType],
        _region_interfaces: &[RegionInterface<DimensionType>],
    ) -> Result<Vec<DimensionType>, TypeError> {
        self.validate_input_types(input_types)?;
        if let DimensionRequirementProof::Disproven(error) = self.prove_from_types() {
            return Err(error.into());
        }
        Ok(Vec::new())
    }

    fn effects(&self) -> Effects {
        match self.prove_from_types() {
            DimensionRequirementProof::Proven => Effects::PURE,
            DimensionRequirementProof::Disproven(_) | DimensionRequirementProof::Inconclusive => {
                Effects::single(Effect::OrderedAssertion)
            }
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self {
            requirement: self.requirement,
            left: self.left.rename_identities(renaming)?,
            right: self.right.as_ref().map(|right| right.rename_identities(renaming)).transpose()?,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self.requirement {
            DimensionRequirement::Bounds(bounds) => OperationFormatter::new(formatter, indentation, self.name())?
                .bracketed(|operation| operation.field("bounds", bounds)),
            _ => formatter.write_str(self.name()),
        }
    }
}

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
        if self.right.is_some() {
            self.infer_output_types(&[inputs[0].r#type().clone(), inputs[1].r#type().clone()], &[])?;
        } else {
            self.infer_output_types(&[inputs[0].r#type().clone()], &[])?;
        }
        self.evaluate(inputs)?;
        Ok(Vec::new())
    }
}

impl<C: Context<Type = DimensionType, Constant = DimensionValue, Operation: From<Self>>>
    PartiallyEvaluatableOperation<C> for DimensionRequirementOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        _driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        match self.prove_from_partial(context, inputs)? {
            DimensionRequirementProof::Proven => Ok(Vec::new()),
            DimensionRequirementProof::Disproven(error) => Err(error.into()),
            DimensionRequirementProof::Inconclusive => context.fold_or_residualize(self.clone(), Vec::new(), inputs),
        }
    }
}

/// Homogeneous operation family for first-class runtime dimensions.
#[derive(Clone, Debug, Operation)]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    /// Dimension literal.
    Constant(ConstantOperation<V>),

    /// Checked binary dimension arithmetic.
    Arithmetic(DimensionArithmeticOperation),

    /// Ordered runtime dimension requirement.
    Requirement(DimensionRequirementOperation),
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, StagingContext};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{Program, ProgramBuilder};
    use crate::tracing::Trace;

    use super::*;

    /// Builds a zero-result program containing one dimension requirement.
    fn requirement_program(
        operation: DimensionRequirementOperation,
        input_types: &[DimensionType],
    ) -> Program<DimensionValue, DimensionOperation<DimensionValue>, Vec<DimensionValue>, Vec<DimensionValue>> {
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let inputs = input_types.iter().cloned().map(|input| builder.add_input(input)).collect::<Vec<_>>();
        builder.add_instruction(operation, Vec::new(), inputs).unwrap();
        builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(
                Vec::new(),
                vec![Placeholder; input_types.len()],
                Vec::new(),
            )
            .unwrap()
    }

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
    fn test_dimension_requirement() {
        let shared = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(10)).unwrap()));
        let equal = DimensionRequirementOperation::equal(&shared, &shared);
        assert_eq!(equal.requirement(), DimensionRequirement::Equal);
        assert_eq!(equal.left_type(), &shared);
        assert_eq!(equal.right_type(), Some(&shared));
        assert_eq!(equal.infer_output_types(&[shared.clone(), shared.clone()], &[]), Ok(Vec::new()));
        assert_eq!(equal.effects(), Effects::PURE);
        assert_eq!(equal.to_string(), DIMENSION_REQUIRE_EQUAL_OPERATION_NAME);

        // Disjoint intervals disprove equality, while overlapping intervals retain an ordered runtime assertion.
        let low = DimensionType::new(DimensionVariable::new("low", DimensionBounds::new(0, Some(4)).unwrap()));
        let high = DimensionType::new(DimensionVariable::new("high", DimensionBounds::new(5, Some(9)).unwrap()));
        let error = DimensionRequirementOperation::equal(&low, &high)
            .infer_output_types(&[low.clone(), high.clone()], &[])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "low == high is impossible from declared bounds".to_string(),
            }),
        );
        let overlapping =
            DimensionType::new(DimensionVariable::new("overlapping", DimensionBounds::new(2, Some(7)).unwrap()));
        let equal = DimensionRequirementOperation::equal(&low, &overlapping);
        assert_eq!(equal.infer_output_types(&[low.clone(), overlapping.clone()], &[]), Ok(Vec::new()),);
        assert_eq!(equal.effects(), Effects::single(Effect::OrderedAssertion));

        // Ordering and divisibility use the same proof lattice without losing their predicate-specific diagnostics.
        let less_than_or_equal = DimensionRequirementOperation::less_than_or_equal(&low, &high);
        assert_eq!(less_than_or_equal.infer_output_types(&[low.clone(), high.clone()], &[]), Ok(Vec::new()));
        assert_eq!(less_than_or_equal.effects(), Effects::PURE);
        let error = DimensionRequirementOperation::less_than_or_equal(&high, &low)
            .infer_output_types(&[high.clone(), low.clone()], &[])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "high <= low is impossible from declared bounds".to_string(),
            }),
        );
        let twelve = DimensionValue::constant(12).unwrap();
        let four = DimensionValue::constant(4).unwrap();
        let divisible = DimensionRequirementOperation::divisible_by(twelve.r#type(), four.r#type());
        assert_eq!(
            divisible.infer_output_types(&[twelve.r#type().clone(), four.r#type().clone()], &[]),
            Ok(Vec::new()),
        );
        assert_eq!(divisible.effects(), Effects::PURE);
        let five = DimensionValue::constant(5).unwrap();
        let error = DimensionRequirementOperation::divisible_by(twelve.r#type(), five.r#type())
            .infer_output_types(&[twelve.r#type().clone(), five.r#type().clone()], &[])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation { message: "12 % 5 == 0; observed 12=12, 5=5".to_string() }),
        );
        let maybe_dividend =
            DimensionType::new(DimensionVariable::new("dividend", DimensionBounds::new(1, Some(13)).unwrap()));
        let maybe_divisor =
            DimensionType::new(DimensionVariable::new("divisor", DimensionBounds::new(2, Some(5)).unwrap()));
        let divisible = DimensionRequirementOperation::divisible_by(&maybe_dividend, &maybe_divisor);
        assert_eq!(divisible.infer_output_types(&[maybe_dividend.clone(), maybe_divisor.clone()], &[]), Ok(Vec::new()),);
        assert_eq!(divisible.effects(), Effects::single(Effect::OrderedAssertion));

        // Explicit bounds are metadata on the requirement, so rendering preserves them even though it has no results.
        let required_bounds = DimensionBounds::new(2, Some(8)).unwrap();
        let bounds = DimensionRequirementOperation::bounds(&overlapping, required_bounds);
        assert_eq!(bounds.requirement(), DimensionRequirement::Bounds(required_bounds));
        assert_eq!(bounds.left_type(), &overlapping);
        assert_eq!(bounds.right_type(), None);
        assert_eq!(bounds.infer_output_types(std::slice::from_ref(&overlapping), &[]), Ok(Vec::new()));
        assert_eq!(bounds.effects(), Effects::PURE);
        assert_eq!(bounds.to_string(), "dimension_require_bounds [bounds=[2, 8)]");
        let disjoint_bounds = DimensionBounds::new(0, Some(4)).unwrap();
        let error = DimensionRequirementOperation::bounds(&high, disjoint_bounds)
            .infer_output_types(std::slice::from_ref(&high), &[])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "high in [0, 4) is impossible from declared bounds".to_string(),
            }),
        );

        // Eager execution reports the declared operand names and concrete observed values.
        let context = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));
        let runtime_bounds = DimensionRequirementOperation::bounds(&left, required_bounds);
        assert_eq!(runtime_bounds.infer_output_types(std::slice::from_ref(&left), &[]), Ok(Vec::new()));
        assert_eq!(runtime_bounds.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            context.bind(
                DimensionRequirementOperation::less_than_or_equal(&left, &right),
                Vec::new(),
                &[DimensionValue::new(left.clone(), 3).unwrap(), DimensionValue::new(right.clone(), 7).unwrap(),],
            ),
            Ok(Vec::new()),
        );
        let error = context
            .bind(
                DimensionRequirementOperation::less_than_or_equal(&left, &right),
                Vec::new(),
                &[DimensionValue::new(left.clone(), 7).unwrap(), DimensionValue::new(right, 3).unwrap()],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left <= right; observed left=7, right=3".to_string(),
            }),
        );
        let error = context.bind(runtime_bounds, Vec::new(), &[DimensionValue::new(left, 9).unwrap()]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "left".to_string(),
                value: 9,
                bounds: required_bounds,
            }),
        );
    }

    #[test]
    fn test_dimension_requirement_effects_and_partial_evaluation() {
        let left = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));

        // Simplification erases a proven pure requirement but preserves inconclusive zero-result assertions in source
        // order, including all of their operand dependencies.
        let mut builder = ProgramBuilder::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left_atom = builder.add_input(left.clone());
        let right_atom = builder.add_input(right.clone());
        builder
            .add_instruction(DimensionRequirementOperation::equal(&left, &left), Vec::new(), vec![left_atom, left_atom])
            .unwrap();
        builder
            .add_instruction(
                DimensionRequirementOperation::less_than_or_equal(&left, &right),
                Vec::new(),
                vec![left_atom, right_atom],
            )
            .unwrap();
        builder
            .add_instruction(
                DimensionRequirementOperation::equal(&left, &right),
                Vec::new(),
                vec![left_atom, right_atom],
            )
            .unwrap();
        let program = builder
            .build::<Vec<DimensionValue>, Vec<DimensionValue>>(Vec::new(), vec![Placeholder, Placeholder], Vec::new())
            .unwrap();
        assert_eq!(program.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(program.instructions().len(), 3);
        let simplified = program.simplified().unwrap();
        assert_eq!(simplified.instructions().len(), 2);
        assert_eq!(
            simplified
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec![DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_EQUAL_OPERATION_NAME],
        );

        // The first failing assertion determines the diagnostic.
        let error = simplified
            .interpret(vec![
                DimensionValue::new(left.clone(), 7).unwrap(),
                DimensionValue::new(right.clone(), 3).unwrap(),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left <= right; observed left=7, right=3".to_string(),
            }),
        );
        let error = simplified
            .interpret(vec![
                DimensionValue::new(left.clone(), 3).unwrap(),
                DimensionValue::new(right.clone(), 7).unwrap(),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left == right; observed left=3, right=7".to_string(),
            }),
        );

        // Unknown operands residualize one ordered assertion, while known operands erase a passing requirement or
        // return its exact observed-value failure on the known side.
        let equality =
            requirement_program(DimensionRequirementOperation::equal(&left, &right), &[left.clone(), right.clone()]);
        let residual = equality
            .partially_evaluate(&[PartialValue::Unknown(left.clone()), PartialValue::Unknown(right.clone())])
            .unwrap();
        assert_eq!(residual.program().instructions().len(), 1);
        assert_eq!(residual.program().effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(residual.inputs().len(), 2);
        let passing = equality
            .partially_evaluate(&[
                PartialValue::Known(DimensionValue::constant(4).unwrap()),
                PartialValue::Known(DimensionValue::constant(4).unwrap()),
            ])
            .unwrap();
        assert!(passing.program().instructions().is_empty());
        let error = equality
            .partially_evaluate(&[
                PartialValue::Known(DimensionValue::constant(4).unwrap()),
                PartialValue::Known(DimensionValue::constant(5).unwrap()),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left == right; observed left=4, right=5".to_string(),
            }),
        );

        // A resolved known operand combines with the unknown operand's bounds, avoiding a redundant residual check.
        let bounded_right =
            DimensionType::new(DimensionVariable::new("bounded_right", DimensionBounds::new(5, Some(10)).unwrap()));
        let ordering = requirement_program(
            DimensionRequirementOperation::less_than_or_equal(&left, &bounded_right),
            &[left.clone(), bounded_right.clone()],
        );
        let proven = ordering
            .partially_evaluate(&[
                PartialValue::Known(DimensionValue::new(left, 2).unwrap()),
                PartialValue::Unknown(bounded_right),
            ])
            .unwrap();
        assert!(proven.program().instructions().is_empty());
        assert!(proven.program().effects().is_pure());
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
