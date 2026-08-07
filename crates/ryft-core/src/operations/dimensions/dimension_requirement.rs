use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};
use crate::contexts::{Context, Domain, ValueResolution};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
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

/// Canonical operation name for an equality [`DimensionRequirementOperation`].
pub const DIMENSION_REQUIRE_EQUAL_OPERATION_NAME: &str = "dimension_require_equal";

/// Canonical operation name for a less-than-or-equal [`DimensionRequirementOperation`].
pub const DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME: &str = "dimension_require_less_than_or_equal";

/// Canonical operation name for a divisibility [`DimensionRequirementOperation`].
pub const DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME: &str = "dimension_require_divisible_by";

/// Canonical operation name for an explicit-bounds [`DimensionRequirementOperation`].
pub const DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME: &str = "dimension_require_bounds";

/// Requirement predicate selected by [`DimensionRequirementOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum DimensionRequirementPredicate {
    /// Requires `left == right`.
    Equal,

    /// Requires `left <= right`.
    LessThanOrEqual,

    /// Requires a positive `divisor` that divides `dividend` exactly.
    DivisibleBy,

    /// Requires one input to lie within the provided bounds.
    Bounds(DimensionBounds),
}

/// Asserts relationships required of first-class runtime dimensions.
///
/// Requirements proven from exact values, shared identities, or interval bounds are pure and may be erased.
/// Statically impossible requirements fail during type inference. Inconclusive requirements remain ordered runtime
/// assertions so that they cannot be eliminated and the first observed failure remains deterministic.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{DimensionRequirement, DimensionValue, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let twelve = DimensionValue::constant(12)?;
/// twelve.require_divisible_by(&DimensionValue::constant(4)?)?;
/// assert!(twelve.require_equal(&DimensionValue::constant(7)?).is_err());
/// # Ok(())
/// # }
/// ```
pub trait DimensionRequirement: Typed<Type = DimensionType> + Sized {
    /// Requires `self == right`.
    fn require_equal(&self, right: &Self) -> Result<(), ProgramError>;

    /// Requires `self <= right`.
    fn require_less_than_or_equal(&self, right: &Self) -> Result<(), ProgramError>;

    /// Requires `right` to be positive and to divide `self` exactly.
    fn require_divisible_by(&self, right: &Self) -> Result<(), ProgramError>;

    /// Requires `self` to lie within `bounds`.
    fn require_bounds(&self, bounds: DimensionBounds) -> Result<(), ProgramError>;
}

impl<V: Value<Type = DimensionType>> DimensionRequirement for V
where
    V::DispatchDomain: Context<Type = DimensionType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionRequirementOperation>,
{
    #[inline]
    fn require_equal(&self, right: &Self) -> Result<(), ProgramError> {
        let operation = DimensionRequirementOperation::equal(&self.r#type(), &right.r#type());
        self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), right.clone()])?;
        Ok(())
    }

    #[inline]
    fn require_less_than_or_equal(&self, right: &Self) -> Result<(), ProgramError> {
        let operation = DimensionRequirementOperation::less_than_or_equal(&self.r#type(), &right.r#type());
        self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), right.clone()])?;
        Ok(())
    }

    #[inline]
    fn require_divisible_by(&self, right: &Self) -> Result<(), ProgramError> {
        let operation = DimensionRequirementOperation::divisible_by(&self.r#type(), &right.r#type());
        self.dispatch_domain().bind(operation, Vec::new(), &[self.clone(), right.clone()])?;
        Ok(())
    }

    #[inline]
    fn require_bounds(&self, bounds: DimensionBounds) -> Result<(), ProgramError> {
        let operation = DimensionRequirementOperation::bounds(&self.r#type(), bounds);
        self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        Ok(())
    }
}

/// Zero-result runtime-dimension assertion used by [`DimensionRequirement`].
///
/// Refer to [`DimensionRequirement`] for semantic details and an example.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionRequirementOperation {
    /// Predicate enforced by this operation.
    predicate: DimensionRequirementPredicate,

    /// Expected left or sole input operand type.
    left: DimensionType,

    /// Expected right operand type for binary predicates.
    right: Option<DimensionType>,
}

impl DimensionRequirementOperation {
    /// Constructs an equality requirement.
    #[inline]
    pub fn equal(left: &DimensionType, right: &DimensionType) -> Self {
        Self { predicate: DimensionRequirementPredicate::Equal, left: left.clone(), right: Some(right.clone()) }
    }

    /// Constructs a less-than-or-equal requirement.
    #[inline]
    pub fn less_than_or_equal(left: &DimensionType, right: &DimensionType) -> Self {
        Self {
            predicate: DimensionRequirementPredicate::LessThanOrEqual,
            left: left.clone(),
            right: Some(right.clone()),
        }
    }

    /// Constructs a positive-divisibility requirement.
    #[inline]
    pub fn divisible_by(dividend: &DimensionType, divisor: &DimensionType) -> Self {
        Self {
            predicate: DimensionRequirementPredicate::DivisibleBy,
            left: dividend.clone(),
            right: Some(divisor.clone()),
        }
    }

    /// Constructs an explicit-bounds requirement.
    #[inline]
    pub fn bounds(input: &DimensionType, bounds: DimensionBounds) -> Self {
        Self { predicate: DimensionRequirementPredicate::Bounds(bounds), left: input.clone(), right: None }
    }

    /// Returns the predicate enforced by this operation.
    #[inline]
    pub fn predicate(&self) -> DimensionRequirementPredicate {
        self.predicate
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

    /// Returns this requirement's operand count.
    #[inline]
    pub(crate) fn input_count(&self) -> usize {
        1 + usize::from(self.right.is_some())
    }

    /// Evaluates this requirement against concrete extents.
    pub(crate) fn evaluate_extents(&self, left: usize, right: Option<usize>) -> Result<(), DimensionError> {
        match self.predicate {
            DimensionRequirementPredicate::Equal => {
                Self::evaluate_equal(&self.left, left, self.right.as_ref().unwrap(), right.unwrap())
            }
            DimensionRequirementPredicate::LessThanOrEqual => {
                Self::evaluate_less_than_or_equal(&self.left, left, self.right.as_ref().unwrap(), right.unwrap())
            }
            DimensionRequirementPredicate::DivisibleBy => {
                Self::evaluate_divisible_by(&self.left, left, self.right.as_ref().unwrap(), right.unwrap())
            }
            DimensionRequirementPredicate::Bounds(bounds) => {
                if bounds.contains(left) {
                    Ok(())
                } else {
                    Err(DimensionError::BindingOutOfBounds {
                        variable: self.left.variable().to_string(),
                        value: left,
                        bounds,
                    })
                }
            }
        }
    }

    /// Returns this requirement's canonical program name.
    fn operation_name(&self) -> &'static str {
        match self.predicate {
            DimensionRequirementPredicate::Equal => DIMENSION_REQUIRE_EQUAL_OPERATION_NAME,
            DimensionRequirementPredicate::LessThanOrEqual => DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME,
            DimensionRequirementPredicate::DivisibleBy => DIMENSION_REQUIRE_DIVISIBLE_BY_OPERATION_NAME,
            DimensionRequirementPredicate::Bounds(_) => DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME,
        }
    }

    /// Proves, disproves, or retains this requirement from type-level facts.
    fn prove_from_types(&self) -> DimensionRequirementProof {
        let left = AbstractDimensionValue::from_type(&self.left);
        match self.predicate {
            DimensionRequirementPredicate::Equal => {
                Self::prove_equal(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirementPredicate::LessThanOrEqual => {
                Self::prove_less_than_or_equal(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirementPredicate::DivisibleBy => {
                Self::prove_divisible_by(&left, &AbstractDimensionValue::from_type(self.right.as_ref().unwrap()))
            }
            DimensionRequirementPredicate::Bounds(bounds) => Self::prove_bounds(&left, bounds),
        }
    }

    /// Proves, disproves, or retains this requirement from partial-evaluation facts.
    fn prove_from_partial<C: Context<Type = DimensionType, Constant: Concretizable<usize>>>(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<DimensionRequirementProof, ProgramError> {
        check_count!("input", inputs, self.input_count(), ProgramError);
        let left = AbstractDimensionValue::from_partial(context, &self.left, &inputs[0]);
        Ok(match self.predicate {
            DimensionRequirementPredicate::Equal => Self::prove_equal(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirementPredicate::LessThanOrEqual => Self::prove_less_than_or_equal(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirementPredicate::DivisibleBy => Self::prove_divisible_by(
                &left,
                &AbstractDimensionValue::from_partial(context, self.right.as_ref().unwrap(), &inputs[1]),
            ),
            DimensionRequirementPredicate::Bounds(bounds) => Self::prove_bounds(&left, bounds),
        })
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
            let left_variable = left_type.variable();
            let right_variable = right_type.variable();
            Err(DimensionError::RequirementViolation {
                message: format!(
                    "{left_variable} == {right_variable}; observed {left_variable}={left}, \
                     {right_variable}={right}",
                ),
            })
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
            let left_variable = left_type.variable();
            let right_variable = right_type.variable();
            Err(DimensionError::RequirementViolation {
                message: format!(
                    "{left_variable} <= {right_variable}; observed {left_variable}={left}, \
                     {right_variable}={right}",
                ),
            })
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
            let dividend_variable = dividend_type.variable();
            let divisor_variable = divisor_type.variable();
            Err(DimensionError::RequirementViolation {
                message: format!(
                    "{divisor_variable} > 0 for divisibility; observed {dividend_variable}={dividend}, \
                     {divisor_variable}={divisor}",
                ),
            })
        } else if dividend.is_multiple_of(divisor) {
            Ok(())
        } else {
            let dividend_variable = dividend_type.variable();
            let divisor_variable = divisor_type.variable();
            Err(DimensionError::RequirementViolation {
                message: format!(
                    "{dividend_variable} % {divisor_variable} == 0; observed {dividend_variable}={dividend}, \
                     {divisor_variable}={divisor}",
                ),
            })
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

impl Operation for DimensionRequirementOperation {
    type Type = DimensionType;

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
            predicate: self.predicate,
            left: self.left.rename_identities(renaming)?,
            right: self.right.as_ref().map(|right| right.rename_identities(renaming)).transpose()?,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self.predicate {
            DimensionRequirementPredicate::Bounds(bounds) => {
                OperationFormatter::new(formatter, indentation, self.name())?
                    .bracketed(|operation| operation.field("bounds", bounds))
            }
            _ => formatter.write_str(self.name()),
        }
    }
}

impl<C: Domain<Type = DimensionType, Value: DimensionRequirement>> InterpretableOperation<C>
    for DimensionRequirementOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, self.input_count(), ProgramError);
        let left_type = inputs[0].r#type().into_owned();
        if let Some(right) = inputs.get(1) {
            self.validate_input_types(&[left_type, right.r#type().into_owned()])?;
        } else {
            self.validate_input_types(std::slice::from_ref(&left_type))?;
        }
        match self.predicate {
            DimensionRequirementPredicate::Equal => inputs[0].require_equal(&inputs[1])?,
            DimensionRequirementPredicate::LessThanOrEqual => {
                inputs[0].require_less_than_or_equal(&inputs[1])?;
            }
            DimensionRequirementPredicate::DivisibleBy => inputs[0].require_divisible_by(&inputs[1])?,
            DimensionRequirementPredicate::Bounds(bounds) => inputs[0].require_bounds(bounds)?,
        }
        Ok(Vec::new())
    }
}

impl<C: Context<Type = DimensionType, Constant: Concretizable<usize>, Operation: From<Self>>>
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

/// Outcome of proving a dimension requirement from facts available at one program point.
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

    /// Constructs an abstract value from partial-evaluation facts.
    fn from_partial<C: Context<Type = DimensionType, Constant: Concretizable<usize>>>(
        context: &PartialEvaluationContext<C>,
        declared_type: &DimensionType,
        value: &PartialEvaluationValue<C::Value>,
    ) -> Self {
        let actual_type = value.r#type();
        let bounds = actual_type.bounds();
        let mut exact =
            bounds.upper().filter(|upper| bounds.lower().checked_add(1) == Some(*upper)).map(|_| bounds.lower());
        if let Some(value) = value.as_known()
            && let ValueResolution::Constant(value) = context.parent().resolve(value)
            && let Ok(extent) = value.concretize()
        {
            exact = Some(extent);
        }
        Self { r#type: declared_type.clone(), bounds, exact }
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DimensionOperation, DimensionValue};
    use crate::contexts::{Context, EagerContext};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{Program, ProgramBuilder};

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
    fn test_dimension_requirement_operation() {
        let shared = DimensionType::new(DimensionVariable::new("shared", DimensionBounds::new(0, Some(10)).unwrap()));
        let equal = DimensionRequirementOperation::equal(&shared, &shared);
        assert_eq!(equal.predicate(), DimensionRequirementPredicate::Equal);
        assert_eq!(equal.left_type(), &shared);
        assert_eq!(equal.right_type(), Some(&shared));
        assert_eq!(equal.infer_output_types(&[shared.clone(), shared.clone()], &[]), Ok(Vec::new()),);
        assert_eq!(equal.effects(), Effects::PURE);
        assert_eq!(equal.to_string(), DIMENSION_REQUIRE_EQUAL_OPERATION_NAME);

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

        let required_bounds = DimensionBounds::new(2, Some(8)).unwrap();
        let bounds = DimensionRequirementOperation::bounds(&overlapping, required_bounds);
        assert_eq!(bounds.predicate(), DimensionRequirementPredicate::Bounds(required_bounds),);
        assert_eq!(bounds.to_string(), "dimension_require_bounds [bounds=[2, 8)]");

        let context = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new();
        let left = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));
        let error = context
            .bind(
                DimensionRequirementOperation::less_than_or_equal(&left, &right),
                Vec::new(),
                &[DimensionValue::new(left, 7).unwrap(), DimensionValue::new(right, 3).unwrap()],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::RequirementViolation {
                message: "left <= right; observed left=7, right=3".to_string(),
            }),
        );
    }

    #[test]
    fn test_dimension_requirement_effects_and_partial_evaluation() {
        let left = DimensionType::new(DimensionVariable::new("left", DimensionBounds::new(0, Some(10)).unwrap()));
        let right = DimensionType::new(DimensionVariable::new("right", DimensionBounds::new(0, Some(10)).unwrap()));

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
        let simplified = program.simplified().unwrap();
        assert_eq!(
            simplified
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec![DIMENSION_REQUIRE_LESS_THAN_OR_EQUAL_OPERATION_NAME, DIMENSION_REQUIRE_EQUAL_OPERATION_NAME,],
        );

        let equality =
            requirement_program(DimensionRequirementOperation::equal(&left, &right), &[left.clone(), right.clone()]);
        let residual = equality
            .partially_evaluate(&[PartialValue::Unknown(left.clone()), PartialValue::Unknown(right.clone())])
            .unwrap();
        assert_eq!(residual.program().instructions().len(), 1);
        assert_eq!(residual.program().effects(), Effects::single(Effect::OrderedAssertion));

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
    }
}
