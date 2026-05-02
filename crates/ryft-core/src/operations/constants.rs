use std::fmt::Display;

use crate::macros::check_input_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, TracingError};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Synthesizes a _zero_ value for a given [`Type`]. [`Zero`] is the [`Type`]-driven counterpart to [`ZeroLike`]; it is
/// what [`ZeroOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait Zero<T: Type>: Sized {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(r#type: &T) -> Result<Self, TracingError>;
}

/// Trait that represents [`Operation`] carrier types that support/include [`ZeroOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`ZeroOperation`] without
/// knowing which carrier is in use.
pub trait SupportsZero<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the zero [`Operation`].
    fn zero_operation(r#type: T) -> Self;

    /// Returns the [`ZeroOperation`] that this operation carrier holds, or `None` if it does not hold one.
    /// Higher-order transformation passes (e.g., the traced reverse-mode pipeline that has to materialize
    /// [`ZeroOperation`] instances into outer-trace constants before its pullback can be interpreted) use this hook
    /// to identify [`ZeroOperation`] instances without pattern-matching on concrete operation carrier types.
    fn as_zero_operation(&self) -> Option<&ZeroOperation<T>> {
        None
    }
}

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for [`ZeroOperation::r#type`]). Note that for arrays, this would typically correspond
/// to an array of the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type = ArrayType> {
    /// [`Type`] of the value produced when this operation is interpreted.
    pub r#type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a new [`ZeroOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
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

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, V: Typed<T> + Zero<T>> InterpretableOperation<T, V> for ZeroOperation<T> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 0, TracingError);
        Ok(vec![V::zero(&self.r#type)?])
    }
}

/// Synthesizes a _one_ value for a given [`Type`]. [`One`] is the [`Type`]-driven counterpart to [`OneLike`]; it is
/// what [`OneOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait One<T: Type>: Sized {
    /// Returns a _one_ value for the provided [`Type`].
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

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for [`OneOperation::r#type`]). Note that for arrays, this would typically correspond
/// to an array of the right type and shape filled with ones.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type = ArrayType> {
    /// [`Type`] of the value produced when this operation is interpreted.
    pub r#type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a new [`OneOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
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

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, V: Typed<T> + One<T>> InterpretableOperation<T, V> for OneOperation<T> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 0, TracingError);
        Ok(vec![V::one(&self.r#type)?])
    }
}
