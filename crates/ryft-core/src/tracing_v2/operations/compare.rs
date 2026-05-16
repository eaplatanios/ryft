use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation};
use crate::tracing::domains::{Tracer, TracingDomain};
use crate::tracing::{Traceable, TracingError};
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

/// Trait that represents [`Operation`] carrier types that support/include [`CompareOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`CompareOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsCompare<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the compare [`Operation`] with the
    /// provided comparison kind.
    fn compare_operation(kind: CompareKind) -> Self;
}

/// Value-level pairwise comparison capability.
///
/// `left.compare(right, kind)` produces an array of `DataType::Boolean` whose `i`-th element is
/// the result of comparing the `i`-th elements of `left` and `right` according to `kind`. Inputs
/// must be broadcast-compatible.
pub trait Compare<Rhs = Self>: Sized {
    /// Compares `self` and `rhs` elementwise using the predicate selected by `kind`.
    fn compare(self, rhs: Rhs, kind: CompareKind) -> Self;
}

impl<'domain, D> Compare for Tracer<'domain, D>
where
    D: TracingDomain<Type = ArrayType>,
    D::OperationCarrier: SupportsCompare<ArrayType, D::Value>,
{
    #[inline]
    fn compare(self, rhs: Self, kind: CompareKind) -> Self {
        self.binary(rhs, D::OperationCarrier::compare_operation(kind))
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
        let input_type_refs: Vec<&ArrayType> = input_types.iter().collect();
        let broadcasted = ArrayType::broadcasted(&input_type_refs).map_err(|_| TypeError {
            message: (format!("{} input types are not broadcast-compatible", ElementwiseOperation::name(self))).into(),
        })?;
        let output_type = ArrayType::new(
            DataType::Boolean,
            broadcasted.shape().clone(),
            broadcasted.layout().cloned(),
            broadcasted.sharding().cloned(),
        )
        .map_err(|error| TypeError { message: (error.to_string()).into() })?;
        Ok(vec![output_type])
    }
}

impl<V: Traceable<ArrayType> + Compare> InterpretableOperation<ArrayType, V> for CompareOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![inputs[0].clone().compare(inputs[1].clone(), self.kind)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

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
        let lhs = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let rhs = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let outputs = CompareOperation::new(CompareKind::Lt).infer_output_types(&[lhs, rhs]).unwrap();
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
