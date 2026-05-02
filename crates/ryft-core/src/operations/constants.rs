use std::fmt::Display;

use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, TracingError};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Synthesizes a typed zero value without an exemplar.
///
/// [`Zero`] is the type-driven counterpart to
/// [`ZeroLike`](crate::tracing_v2::operations::constants::ZeroLike): it is what [`ZeroOperation`] needs in order to
/// evaluate at interpretation time, since the op carries only the output type and has no input values to derive a shape
/// from. Concrete leaf value types implement this trait directly.
///
/// Wrapper types that fundamentally cannot synthesize a zero from metadata alone should use exemplar-backed
/// [`ZeroLike`](crate::tracing_v2::operations::constants::ZeroLike) where possible. Programs containing `Zero` ops over
/// those value types must materialize them away before being interpreted.
pub trait Zero<T: Type>: Sized {
    /// Returns a typed zero whose shape and dtype are described by `r#type`.
    fn zero(r#type: &T) -> Result<Self, TracingError>;
}

/// Trait that represents [`Operation`] carrier types that support/include [`ZeroOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`ZeroOperation`] without
/// knowing which carrier is in use.
pub trait SupportsZero<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the zero [`Operation`].
    fn zero_operation(r#type: T) -> Self;

    /// Returns the zero [`Operation`], or `None` for any other operation variant.
    ///
    /// Higher-order passes (notably the traced reverse-mode pipeline that has to materialize `Zero` ops into
    /// outer-trace constants before its pullback can be interpreted) use this hook to identify zero ops without
    /// pattern-matching on a concrete carrier enum.
    fn as_zero_operation(&self) -> Option<&ZeroOperation<T>> {
        None
    }
}

/// Typed-zero primitive: a 0-input, 1-output op that produces a value of the carried type metadata.
///
/// [`ZeroOperation`] is emitted by the linear-program transpose pass at the pullback boundary for primal inputs that
/// have no cotangent contribution accumulated onto them. Closed carriers implement [`SupportsZero`] to construct the
/// carrier-specific representation, and the carrier's own trait impls then delegate to this op for [`Operation`] and
/// [`InterpretableOperation`] semantics.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type = ArrayType> {
    /// Type of the value produced when this op is interpreted.
    pub output_type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a zero op that produces values of `output_type`.
    #[inline]
    pub fn new(output_type: T) -> Self {
        Self { output_type }
    }
}

impl<T: Type> Display for ZeroOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: Type> Operation<T> for ZeroOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        "zero"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if !input_types.is_empty() {
            return Err(TypeError { message: format!("zero expected 0 input types but got {}", input_types.len()) });
        }
        Ok(vec![self.output_type.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("type", &self.output_type))
    }
}

impl<T: Type, V: Typed<T> + Zero<T>> InterpretableOperation<T, V> for ZeroOperation<T> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if !inputs.is_empty() {
            return Err(TracingError::InvalidInputCount { expected: 0, got: inputs.len() });
        }
        Ok(vec![V::zero(&self.output_type)?])
    }
}

/// Synthesizes a typed unit cotangent seed without an exemplar.
///
/// [`One`] is the seed counterpart to [`Zero`]. It is intentionally fallible because not every abstract descriptor
/// admits the unit seed required by scalar-output reverse-mode transforms. For example, the built-in [`ArrayType`]
/// implementations reject non-rank-0 descriptors so `grad` keeps its scalar-output semantics even though the check
/// depends on runtime metadata.
pub trait One<T: Type>: Sized {
    /// Returns the unit cotangent seed described by `r#type`.
    fn one(r#type: &T) -> Result<Self, TracingError>;
}

/// Trait that represents [`Operation`] carrier types that support/include [`OneOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`OneOperation`] without
/// knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsOne<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the one [`Operation`].
    fn one_operation(r#type: T) -> Self;
}

/// Typed-one primitive: a 0-input, 1-output op that produces a value of the carried type metadata.
///
/// [`OneOperation`] is the staged form of [`One::one`]. It is used for unit cotangent seeds and any other type-driven
/// multiplicative identity where no exemplar value is available.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type = ArrayType> {
    /// Type of the value produced when this op is interpreted.
    pub output_type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a one op that produces values of `output_type`.
    #[inline]
    pub fn new(output_type: T) -> Self {
        Self { output_type }
    }
}

impl<T: Type> Display for OneOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: Type> Operation<T> for OneOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        "one"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if !input_types.is_empty() {
            return Err(TypeError { message: format!("one expected 0 input types but got {}", input_types.len()) });
        }
        Ok(vec![self.output_type.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("type", &self.output_type))
    }
}

impl<T: Type, V: Typed<T> + One<T>> InterpretableOperation<T, V> for OneOperation<T> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if !inputs.is_empty() {
            return Err(TracingError::InvalidInputCount { expected: 0, got: inputs.len() });
        }
        Ok(vec![V::one(&self.output_type)?])
    }
}
