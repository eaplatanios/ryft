use std::fmt::Display;

use half::{bf16, f16};

use crate::broadcasting::Broadcastable;
use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Type, TypeError};

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
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl<T: Type + Broadcastable + BooleanLike> Operation<T> for CompareOperation {
    #[inline]
    fn name(&self) -> &'static str {
        COMPARE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let broadcasted = T::broadcasted(input_types)
            .map_err(|_| TypeError { message: "comparison input types are not broadcast-compatible".to_string() })?;
        Ok(vec![broadcasted.as_boolean()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, COMPARE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("direction", &self.direction))
    }
}

impl<T: Type, V: Value<T> + Compare<Output = V>> InterpretableOperation<T, V> for CompareOperation
where
    Self: Operation<T>,
{
    fn interpret(
        &self,
        _context: &mut <V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].compare(&inputs[1], self.direction)])
    }
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
    fn compare(&self, rhs: &Rhs, direction: ComparisonDirection) -> Self::Output;

    /// Computes `self == rhs` using [`CompareOperation`].
    #[inline]
    fn equal(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::Equal)
    }

    /// Computes `self != rhs` using [`CompareOperation`].
    #[inline]
    fn not_equal(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::NotEqual)
    }

    /// Computes `self < rhs` using [`CompareOperation`].
    #[inline]
    fn less_than(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::LessThan)
    }

    /// Computes `self <= rhs` using [`CompareOperation`].
    #[inline]
    fn less_than_or_equal(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::LessThanOrEqual)
    }

    /// Computes `self > rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::GreaterThan)
    }

    /// Computes `self >= rhs` using [`CompareOperation`].
    #[inline]
    fn greater_than_or_equal(&self, rhs: &Rhs) -> Self::Output {
        self.compare(rhs, ComparisonDirection::GreaterThanOrEqual)
    }
}

macro_rules! impl_compare_for_scalar {
    ($($type:ty => ($zero:expr, $one:expr)),* $(,)?) => {
        $(
            impl Compare for $type {
                type Output = Self;

                #[inline]
                fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Self::Output {
                    let result = match direction {
                        ComparisonDirection::Equal => self == rhs,
                        ComparisonDirection::NotEqual => self != rhs,
                        ComparisonDirection::LessThan => self < rhs,
                        ComparisonDirection::LessThanOrEqual => self <= rhs,
                        ComparisonDirection::GreaterThan => self > rhs,
                        ComparisonDirection::GreaterThanOrEqual => self >= rhs,
                    };
                    if result { $one } else { $zero }
                }
            }
        )*
    };
}

impl_compare_for_scalar!(
    bf16 => (bf16::ZERO, bf16::ONE),
    f16 => (f16::ZERO, f16::ONE),
    f32 => (0.0, 1.0),
    f64 => (0.0, 1.0),
);

impl<C: StagingContext<Operation: From<CompareOperation>>> Compare for Tracer<C> {
    type Output = Self;

    #[inline]
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Self::Output {
        self.binary(rhs, CompareOperation::new(direction))
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

        // Test using `DataType`s: broadcast-compatible (promotable) input data types infer a Boolean output and
        // non-promotable ones error.
        let operation = CompareOperation::new(ComparisonDirection::LessThan);
        assert_eq!(operation.infer_output_types(&[DataType::F64, DataType::F64]), Ok(vec![DataType::Boolean]));
        assert_eq!(operation.infer_output_types(&[DataType::F32, DataType::F64]), Ok(vec![DataType::Boolean]));
        assert_eq!(
            operation.infer_output_types(&[DataType::F8E3M4, DataType::F32]),
            Err(TypeError { message: "comparison input types are not broadcast-compatible".to_string() }),
        );

        // Test that `as_boolean` on type metadata produces Boolean counterparts while `boolean` errors because
        // type metadata carries no concrete payload to decode.
        assert_eq!(DataType::F64.as_boolean(), DataType::Boolean);
        assert!(DataType::F64.boolean().is_err());
        let array_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        assert_eq!(array_type.as_boolean(), ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])));
        assert!(array_type.boolean().is_err());

        // Test that scalar values use the in-band zero/one Boolean encoding.
        assert_eq!(2.0f64.less_than(&3.0), 1.0);
        assert_eq!(2.0f32.greater_than(&3.0), 0.0);
        assert_eq!(
            <CompareOperation as InterpretableOperation<DataType, f64>>::interpret(
                &operation,
                &mut (),
                &[2.0f64, 3.0f64]
            ),
            Ok(vec![1.0])
        );

        // Test using `TestArray`s.
        let lhs = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = TestArray::vector(vec![2.0, 2.0, 2.0, 2.0]);
        let outputs = CompareOperation::new(ComparisonDirection::LessThan).interpret(&mut (), &[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].values(), &[1.0, 0.0, 0.0, 0.0]);

        // Test the convenience functions provided by `Compare`.
        let left = || TestArray::vector(vec![1.0, 2.0, 3.0]);
        let right = || TestArray::vector(vec![2.0, 2.0, 2.0]);
        assert_eq!(left().equal(&right()).values(), &[0.0, 1.0, 0.0]);
        assert_eq!(left().not_equal(&right()).values(), &[1.0, 0.0, 1.0]);
        assert_eq!(left().less_than(&right()).values(), &[1.0, 0.0, 0.0]);
        assert_eq!(left().less_than_or_equal(&right()).values(), &[1.0, 1.0, 0.0]);
        assert_eq!(left().greater_than(&right()).values(), &[0.0, 0.0, 1.0]);
        assert_eq!(left().greater_than_or_equal(&right()).values(), &[0.0, 1.0, 1.0]);
    }
}
