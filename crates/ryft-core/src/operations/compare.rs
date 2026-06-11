use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Canonical operation name for [`CompareOperation`].
pub const COMPARE_OPERATION_NAME: &'static str = "compare";

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
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for CompareOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let broadcasted = ArrayType::broadcasted(input_types)
            .map_err(|_| TypeError { message: "comparison input types are not broadcast-compatible".to_string() })?;
        let output_type = ArrayType::new(DataType::Boolean, broadcasted.shape().clone())
            .with_layout(broadcasted.layout().cloned())
            .with_sharding(broadcasted.sharding().cloned())
            .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![output_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("direction", &self.direction))
    }
}

impl<V: Value<ArrayType> + Compare<Output = V>> InterpretableOperation<ArrayType, V> for CompareOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone().compare(inputs[1].clone(), self.direction)])
    }
}

/// Trait that represents [`Operation`] types that support/include [`CompareOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`CompareOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsCompare<T: Type> {
    /// Constructs the backend-specific representation of the compare [`Operation`] with the
    /// provided [`ComparisonDirection`].
    fn compare_operation(direction: ComparisonDirection) -> Self;
}

/// Represents the ability to perform a pairwise comparison between two values. For array values,
/// `left.compare(right, direction)` produces a Boolean-valued result whose `i`-th element is the result of comparing
/// the `i`-th elements of `left` and `right` according to `direction`. The input arrays must be broadcast-compatible,
/// in this case, and they must also have the same [`DataType`]. For this example, the output has [`DataType::Boolean`]
/// and the broadcasted shape of the two input arrays.
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
    fn compare(self, rhs: Rhs, direction: ComparisonDirection) -> Self::Output;
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsCompare<ArrayType>>> Compare for Tracer<C> {
    type Output = Self;

    #[inline]
    fn compare(self, rhs: Self, direction: ComparisonDirection) -> Self::Output {
        self.binary(rhs, C::Operation::compare_operation(direction))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[test]
    fn test_compare() {
        // Test using `ArrayType`s.
        let lhs = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let rhs = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let outputs =
            CompareOperation::infer_output_types(&CompareOperation::new(ComparisonDirection::LessThan), &[lhs, rhs])
                .unwrap();
        assert_eq!(
            outputs,
            vec![ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2), Size::Static(3)]))]
        );

        // Test using `TestArray`s.
        let lhs = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = TestArray::vector(vec![2.0, 2.0, 2.0, 2.0]);
        let outputs = CompareOperation::new(ComparisonDirection::LessThan).interpret(&[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].values(), &[1.0, 0.0, 0.0, 0.0]);
    }
}
