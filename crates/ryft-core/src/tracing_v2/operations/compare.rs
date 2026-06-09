use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError};

/// Kind of pairwise comparison performed by a [`CompareOperation`].
///
/// Each kind corresponds to one comparison predicate. Inputs must be broadcast-compatible and
/// share a numeric data type; the output has [`DataType::Boolean`] and the broadcasted shape of
/// the inputs. Lowers to StableHLO's `stablehlo.compare` op with the matching predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompareKind {
    /// Elementwise equality: `left == right`.
    Eq,

    /// Elementwise inequality: `left != right`.
    Ne,

    /// Elementwise less-than: `left < right`.
    Lt,

    /// Elementwise less-than-or-equal: `left <= right`.
    Le,

    /// Elementwise greater-than: `left > right`.
    Gt,

    /// Elementwise greater-than-or-equal: `left >= right`.
    Ge,
}

impl CompareKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }
}

impl Display for CompareKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Trait for operation types that include or can wrap [`CompareOperation`].
/// Backend-owned closed operation enums (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`CompareOperation`] without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsCompare<T: Type> {
    /// Constructs the backend-specific representation of the compare [`Operation`] with the
    /// provided comparison kind.
    fn compare_operation(kind: CompareKind) -> Self;
}

/// Value-level pairwise comparison capability.
///
/// `left.compare(right, kind)` produces a Boolean-valued result whose `i`-th element is the
/// result of comparing the `i`-th elements of `left` and `right` according to `kind`. Inputs
/// must be broadcast-compatible.
///
/// The associated [`Output`](Compare::Output) type lets concrete backends choose how they
/// represent Boolean results:
///   - In-band encoding (`Output = Self`): keep the input element type and encode bools as
///     `T::zero()` / `T::one()`. This is what `TestArray` and `ShardMapTensor` do, since the
///     operation-level dispatch (via [`CompareOperation`]) requires the result type to match the
///     input type.
///   - True Boolean representation (`Output = Array<bool>`-like): produce a dedicated Boolean
///     value type. This is what an ndarray backend with separate `Array<bool>` may want for
///     direct user calls, even though the staged operation path still uses in-band encoding.
pub trait Compare<Rhs = Self>: Sized {
    /// Result type of the comparison.
    type Output;

    /// Compares `self` and `rhs` elementwise using the predicate selected by `kind`.
    fn compare(self, rhs: Rhs, kind: CompareKind) -> Self::Output;
}

impl<C> Compare for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsCompare<ArrayType>,
{
    type Output = Self;

    #[inline]
    fn compare(self, rhs: Self, kind: CompareKind) -> Self::Output {
        self.binary(rhs, C::Operation::compare_operation(kind))
    }
}

/// Primitive representing one elementwise pairwise comparison.
///
/// [`CompareOperation`] compares two broadcast-compatible array operands and returns a Boolean
/// array of the broadcasted shape. The semantics of the comparison are described by
/// [`kind`](Self::kind). Lowers to StableHLO's `stablehlo.compare` op in the XLA backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompareOperation {
    /// Kind of comparison.
    kind: CompareKind,
}

impl CompareOperation {
    /// Creates a new [`CompareOperation`] with the supplied kind.
    #[inline]
    pub fn new(kind: CompareKind) -> Self {
        Self { kind }
    }

    /// Returns the kind of comparison.
    #[inline]
    pub fn kind(&self) -> CompareKind {
        self.kind
    }
}

impl Display for CompareOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "compare_{}", self.kind)
    }
}

impl ElementwiseOperation for CompareOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            CompareKind::Eq => "compare_eq",
            CompareKind::Ne => "compare_ne",
            CompareKind::Lt => "compare_lt",
            CompareKind::Le => "compare_le",
            CompareKind::Gt => "compare_gt",
            CompareKind::Ge => "compare_ge",
        }
    }

    #[inline]
    fn input_count(&self) -> usize {
        2
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let broadcasted = ArrayType::broadcasted(input_types).map_err(|_| TypeError {
            message: format!("{} input types are not broadcast-compatible", ElementwiseOperation::name(self)),
        })?;
        let output_type = ArrayType::new(
            DataType::Boolean,
            broadcasted.shape().clone(),
            broadcasted.layout().cloned(),
            broadcasted.sharding().cloned(),
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![output_type])
    }
}

impl<V: Value<ArrayType> + Compare<Output = V>> InterpretableOperation<ArrayType, V> for CompareOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].clone().compare(inputs[1].clone(), self.kind)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
    use crate::tracing_v2::test_util::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    fn boolean_array_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(
            DataType::Boolean,
            Shape::new(dimensions.iter().copied().map(Size::Static).collect()),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_compare_operation_infers_boolean_output_type() {
        let lhs =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let rhs =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let outputs = <CompareOperation as Operation<ArrayType>>::infer_output_types(
            &CompareOperation::new(CompareKind::Lt),
            &[lhs, rhs],
        )
        .unwrap();
        assert_eq!(outputs, vec![boolean_array_type(&[2, 3])]);
    }

    #[test]
    fn test_compare_operation_interprets_lt_on_test_arrays() {
        let lhs = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = TestArray::vector(vec![2.0, 2.0, 2.0, 2.0]);
        let outputs = CompareOperation::new(CompareKind::Lt).interpret(&[lhs, rhs]).unwrap();
        // TestArray uses f64 for everything including bools (0.0 = false, 1.0 = true).
        assert_eq!(outputs[0].values(), &[1.0, 0.0, 0.0, 0.0]);
    }
}
