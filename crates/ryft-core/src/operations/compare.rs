use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::operations::manipulation::conversion::ElementType;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`CompareOperation`].
pub const COMPARE_OPERATION_NAME: &str = "compare";

/// Direction of the pairwise comparison performed by a [`CompareOperation`]. Each direction corresponds to one
/// comparison predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonDirection {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl Display for ComparisonDirection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Equal => "Equal",
            Self::NotEqual => "NotEqual",
            Self::LessThan => "LessThan",
            Self::LessThanOrEqual => "LessThanOrEqual",
            Self::GreaterThan => "GreaterThan",
            Self::GreaterThanOrEqual => "GreaterThanOrEqual",
        })
    }
}

/// [`Operation`] that performs pairwise comparisons. The semantics of the comparison
/// are described by [`direction`](Self::direction).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompareOperation {
    /// [`ComparisonDirection`] used by this [`CompareOperation`].
    direction: ComparisonDirection,
}

impl CompareOperation {
    /// Creates a new [`CompareOperation`] with the provided [`ComparisonDirection`].
    #[inline]
    pub fn new(direction: ComparisonDirection) -> Self {
        Self { direction }
    }

    /// Returns the [`ComparisonDirection`] used by this [`CompareOperation`].
    #[inline]
    pub fn direction(&self) -> ComparisonDirection {
        self.direction
    }
}

impl Display for CompareOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl<T: Broadcastable + ElementType> Operation<T> for CompareOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 2, TypeError);

        // Complex operands are unordered, so only the equality comparison directions are defined for them.
        if !matches!(self.direction, ComparisonDirection::Equal | ComparisonDirection::NotEqual)
            && input_types.iter().any(|input_type| input_type.is_complex())
        {
            return Err(TypeError {
                message: format!(
                    "cannot apply an ordered comparison to unordered complex operands of types {} and {}",
                    input_types[0], input_types[1],
                ),
            });
        }

        let broadcasted = T::broadcasted(input_types)
            .map_err(|_| TypeError { message: "comparison input types are not broadcast-compatible".to_string() })?;
        Ok(vec![broadcasted.with_element_type(DataType::Boolean)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", &self.direction))
    }
}

impl ElementwiseOperation for CompareOperation {
    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Operation::<ArrayType>::infer_output_types(self, input_types, &[])
    }
}

impl<C: Domain> InterpretableOperation<C> for CompareOperation
where
    C::Value: Compare<Output = C::Value>,
    C::Type: Broadcastable + ElementType,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].compare(&inputs[1], self.direction)?])
    }
}

impl<C: Context<Operation: From<CompareOperation>>> PartiallyEvaluatableOperation<C> for CompareOperation {}

impl_differentiable_elementwise_operation!(@non_differentiable CompareOperation);

/// Represents the ability to perform a pairwise comparison between two values. For array values,
/// `left.compare(right, direction)` produces a Boolean-valued result whose `i`-th element is the result of comparing
/// the `i`-th elements of `left` and `right` according to `direction`. The input arrays must be broadcast-compatible,
/// in this case, and they must also have the same [`DataType`](crate::DataType). For this example, the output
/// has [`DataType::Boolean`](crate::DataType::Boolean) and the broadcasted shape of the two input arrays.
///
/// The associated [`Output`](Compare::Output) type lets concrete backends choose how they represent Boolean results:
///
/// - **In-Band Encoding (i.e., `Output = Self`):** Keep the input element type and encode Boolean values
///   as `T::zero()` or `T::one()`.
/// - **True Boolean Representation (i.e., `Output = Array<bool>`-like):** Produce a dedicated Boolean value type.
///   This is what an `ndarray` backend with a separate `Array<bool>` may want for direct user calls, even though the
///   staged operation path still uses in-band encoding.
pub trait Compare<Rhs = Self>: Sized {
    /// Result type of the comparison.
    type Output;

    /// Compares `self` and `rhs` using a predicate determined by the provided `direction`.
    fn compare(&self, rhs: &Rhs, direction: ComparisonDirection) -> Result<Self::Output, ProgramError>;

    /// Computes `self == rhs` using [`CompareOperation`].
    #[inline]
    fn equal(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::Equal)
    }

    /// Computes `self != rhs` using [`CompareOperation`].
    #[inline]
    fn not_equal(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::NotEqual)
    }

    /// Computes `self < rhs` using [`CompareOperation`].
    #[inline]
    fn less_than(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::LessThan)
    }

    /// Computes `self <= rhs` using [`CompareOperation`].
    #[inline]
    fn less_than_or_equal(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::LessThanOrEqual)
    }

    /// Computes `self > rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::GreaterThan)
    }

    /// Computes `self >= rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than_or_equal(&self, rhs: &Rhs) -> Result<Self::Output, ProgramError> {
        self.compare(rhs, ComparisonDirection::GreaterThanOrEqual)
    }
}

impl<V: Value<DispatchDomain: Context<Operation: From<CompareOperation>>>> Compare for V {
    type Output = Self;

    #[inline]
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(CompareOperation::new(direction), Vec::new(), &[self.clone(), rhs.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{DifferentiationTracer, ForwardModeDifferentiate};
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::programs::ProgramError;
    use crate::types::DataType;

    use super::*;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over JVP duals of the eager [`Array`] context.
    fn piecewise_select(
        x: DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>,
    ) -> Result<DifferentiationTracer<EagerContext<Array, ArrayOperation<Array>>>, ProgramError> {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan)?;
        Select::select(&mask, &(x.clone() + x.clone()), &(x.clone() + x.clone() + x))
    }

    #[test]
    fn test_compare() {
        assert_eq!(Scalar::from(2.0).less_than(&Scalar::from(3.0)).unwrap(), Scalar::from(true));
        assert_eq!(Scalar::from(2.0f32).greater_than(&Scalar::from(3.0f32)).unwrap(), Scalar::from(false));
        let left = || Array::vector(vec![1.0, 2.0, 3.0]);
        let right = || Array::vector(vec![2.0, 2.0, 2.0]);
        assert_eq!(left().equal(&right()).unwrap().values(), &[false, true, false]);
        assert_eq!(left().not_equal(&right()).unwrap().values(), &[true, false, true]);
        assert_eq!(left().less_than(&right()).unwrap().values(), &[true, false, false]);
        assert_eq!(left().less_than_or_equal(&right()).unwrap().values(), &[true, true, false]);
        assert_eq!(left().greater_than(&right()).unwrap().values(), &[false, false, true]);
        assert_eq!(left().greater_than_or_equal(&right()).unwrap().values(), &[false, true, true]);
    }

    #[test]
    fn test_compare_type_inference() {
        check_operation_type_inference!(
            @elementwise @binary,
            operation = CompareOperation::new(ComparisonDirection::LessThan),
            cases = [
                {
                    input_data_types = [DataType::F32, DataType::F64],
                    output_data_types = [DataType::Boolean],
                },
                {
                    input_data_types = [DataType::F8E3M4, DataType::F32],
                    error = "comparison input types are not broadcast-compatible",
                },
                {
                    input_data_types = [DataType::C64, DataType::C64],
                    error = "cannot apply an ordered comparison to unordered complex operands of types c64 and c64",
                },
            ],
        );
        check_operation_type_inference!(
            @elementwise @binary,
            operation = CompareOperation::new(ComparisonDirection::Equal),
            cases = [{
                input_data_types = [DataType::C64, DataType::C64],
                output_data_types = [DataType::Boolean],
            }],
        );
    }

    #[test]
    fn test_compare_batching() {
        check_operation_batching!(
            @exact,
            operation = CompareOperation::new(ComparisonDirection::GreaterThan),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                        (@replicated, Array::scalar(0.0)),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![true, false]))],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0)),
                        (@mapped(axis = 0), Array::vector(vec![1.0, -2.0])),
                    ],
                    outputs = [(@mapped(axis = 0), Array::vector(vec![false, true]))],
                },
            ],
        );
    }

    #[test]
    fn test_compare_differentiation() {
        // `f(x) = select(x > 0, 2x, 3x)`: the comparison output is Boolean, so its tangent is symbolically zero and
        // the derivative comes entirely from the selected branch (2 for x > 0 and 3 for x <= 0).
        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(piecewise_select, Array::scalar(2.0), Array::scalar(1.0))
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![2.0]);

        let (primal, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(piecewise_select, Array::scalar(-2.0), Array::scalar(1.0))
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![-6.0]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_compare_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CompareOperation::new(ComparisonDirection::GreaterThan),
            inputs = [Scalar::from(1.0), Scalar::from(0.0)],
            expected = Scalar::from(true),
        );
    }
}
