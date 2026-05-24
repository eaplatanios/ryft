use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::types::{ArrayType, Type};

/// Kind of elementwise logical operation performed by a [`LogicalOperation`].
///
/// Each kind corresponds to one logical operator on Boolean arrays. Inputs are expected to have
/// [`DataType::Boolean`](crate::types::DataType::Boolean); the output also has Boolean data type
/// and the broadcasted shape of the inputs. Lowers to StableHLO's `stablehlo.{and,or,xor,not}` op.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicalKind {
    /// Elementwise logical conjunction (`left & right`). Binary.
    And,

    /// Elementwise logical disjunction (`left | right`). Binary.
    Or,

    /// Elementwise logical exclusive-OR (`left ^ right`). Binary.
    Xor,

    /// Elementwise logical negation (`!input`). Unary.
    Not,
}

impl LogicalKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Not => "not",
        }
    }

    /// Returns the number of input operands consumed by this logical kind.
    pub fn input_count(self) -> usize {
        match self {
            Self::And | Self::Or | Self::Xor => 2,
            Self::Not => 1,
        }
    }
}

impl Display for LogicalKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`LogicalOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`LogicalOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsLogical<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the logical [`Operation`] with the
    /// provided kind.
    fn logical_operation(kind: LogicalKind) -> Self;
}

/// Value-level binary logical capability for [`LogicalKind::And`], [`LogicalKind::Or`], and
/// [`LogicalKind::Xor`].
pub trait LogicalBinary<Rhs = Self>: Sized {
    /// Combines `self` and `rhs` using the binary logical operator selected by `kind`.
    fn logical_binary(self, rhs: Rhs, kind: LogicalKind) -> Self;
}

/// Value-level unary logical-not capability for [`LogicalKind::Not`].
pub trait LogicalNot: Sized {
    /// Computes elementwise `!self`.
    fn logical_not(self) -> Self;
}

impl<C> LogicalBinary for Tracer<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: SupportsLogical<ArrayType, C::Value>,
{
    #[inline]
    fn logical_binary(self, rhs: Self, kind: LogicalKind) -> Self {
        assert!(kind != LogicalKind::Not, "LogicalKind::Not is a unary operation; use LogicalNot::logical_not");
        self.binary(rhs, C::Operation::logical_operation(kind))
    }
}

impl<C> LogicalNot for Tracer<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: SupportsLogical<ArrayType, C::Value>,
{
    #[inline]
    fn logical_not(self) -> Self {
        self.unary(C::Operation::logical_operation(LogicalKind::Not))
    }
}

/// Primitive representing one elementwise logical operation.
///
/// [`LogicalOperation`] applies a Boolean operator described by [`kind`](Self::kind) to its
/// operands. Binary kinds (`And`/`Or`/`Xor`) consume two broadcast-compatible Boolean arrays
/// and return a Boolean array of the broadcasted shape. The unary kind (`Not`) consumes one
/// Boolean array and returns a Boolean array of the same shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalOperation {
    /// Kind of logical operation.
    kind: LogicalKind,
}

impl LogicalOperation {
    /// Creates a new [`LogicalOperation`] with the supplied kind.
    #[inline]
    pub fn new(kind: LogicalKind) -> Self {
        Self { kind }
    }

    /// Returns the kind of logical operation.
    #[inline]
    pub fn kind(&self) -> LogicalKind {
        self.kind
    }
}

impl Display for LogicalOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "logical_{}", self.kind)
    }
}

impl ElementwiseOperation for LogicalOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            LogicalKind::And => "logical_and",
            LogicalKind::Or => "logical_or",
            LogicalKind::Xor => "logical_xor",
            LogicalKind::Not => "logical_not",
        }
    }

    #[inline]
    fn input_count(&self) -> usize {
        self.kind.input_count()
    }
}

impl<V: Traceable<ArrayType> + LogicalBinary + LogicalNot> InterpretableOperation<ArrayType, V> for LogicalOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        match self.kind {
            LogicalKind::Not => {
                check_count!("input", inputs, 1, TracingError);
                Ok(vec![inputs[0].clone().logical_not()])
            }
            LogicalKind::And | LogicalKind::Or | LogicalKind::Xor => {
                check_count!("input", inputs, 2, TracingError);
                Ok(vec![inputs[0].clone().logical_binary(inputs[1].clone(), self.kind)])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::test_util::TestArray;

    use super::*;

    #[test]
    fn test_logical_operation_interprets_and_on_test_arrays() {
        let lhs = TestArray::vector(vec![1.0, 1.0, 0.0, 0.0]);
        let rhs = TestArray::vector(vec![1.0, 0.0, 1.0, 0.0]);
        let outputs = LogicalOperation::new(LogicalKind::And).interpret(&[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].values(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_logical_operation_interprets_not_on_test_array() {
        let input = TestArray::vector(vec![1.0, 0.0, 1.0]);
        let outputs = LogicalOperation::new(LogicalKind::Not).interpret(&[input]).unwrap();
        assert_eq!(outputs[0].values(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_logical_operation_kind_input_count_matches_signature() {
        assert_eq!(LogicalKind::And.input_count(), 2);
        assert_eq!(LogicalKind::Or.input_count(), 2);
        assert_eq!(LogicalKind::Xor.input_count(), 2);
        assert_eq!(LogicalKind::Not.input_count(), 1);
    }
}
