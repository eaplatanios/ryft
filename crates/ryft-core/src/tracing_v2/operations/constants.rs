use std::fmt::{Debug, Display};

use half::{bf16, f16};

use crate::{
    tracing::{AtomId, OperationFormatter, Traceable, TracingError, Value},
    tracing_v2::operations::primitive::LinearPrimitiveOperation,
    types::{ArrayType, Type, TypeError, Typed},
};

use super::{InterpretableOperation, LinearOperation, Operation, TranspositionContext};

/// Returns a zero value with the same structure as an existing value.
///
/// [`ZeroLike`] is the local, value-level counterpart to
/// [`Engine::zero`](crate::tracing_v2::Engine::zero). When a transform already has an exemplar in
/// hand, it uses this trait instead of going back through abstract metadata. That is especially
/// important for wrappers like [`Tracer`](crate::tracing_v2::Tracer) and
/// [`JvpTracer`](crate::tracing_v2::JvpTracer), which can stage or derive a zero from their
/// existing state even when abstract synthesis alone would be insufficient. This module also ships
/// the built-in scalar implementations of [`Traceable`](crate::tracing::Traceable),
/// [`Value`](crate::tracing::Value), [`ZeroLike`], and [`OneLike`].
pub trait ZeroLike {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Returns a one value with the same structure as an existing value.
///
/// This mirrors [`ZeroLike`] for the multiplicative identity. It is used in the same places where
/// transforms need a unit seed from an exemplar, such as reverse-mode pullbacks for scalar-output
/// functions.
pub trait OneLike {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

/// Synthesizes a typed zero value without an exemplar.
///
/// [`Zero`] is the type-driven counterpart to [`ZeroLike`]: it is what [`ZeroOperation`] needs in
/// order to evaluate at interpretation time, since the op carries only the output type and has no
/// input values to derive a shape from. Concrete leaf value types implement this trait directly.
///
/// Wrapper types that fundamentally cannot synthesize a zero from metadata alone (notably
/// [`Tracer`](crate::tracing_v2::Tracer)) implement it as a runtime error. Programs containing
/// `Zero` ops over those value types must materialize them away before being interpreted.
pub trait Zero<T: Type>: Sized {
    /// Returns a typed zero whose shape and dtype are described by `r#type`.
    fn zero(r#type: &T) -> Result<Self, TracingError>;
}

/// Hidden carrier capability for staging the zero primitive.
///
/// `SupportsZero` lets generic transform code stage the typed-zero primitive on linear-program
/// carriers without knowing the carrier type. The transpose pass uses it at the boundary to emit
/// structural zeros for primal inputs that have no contribution accumulated onto them.
#[doc(hidden)]
pub trait SupportsZero<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the typed-zero primitive.
    fn zero_operation(r#type: T) -> Self;

    /// Returns the type carried by a zero op, or `None` for any other op variant.
    ///
    /// Higher-order passes (notably the traced reverse-mode pipeline that has to materialize
    /// `Zero` ops into outer-trace constants before its pullback can be interpreted) use this hook
    /// to identify zero ops without pattern-matching on a concrete carrier enum.
    fn as_zero(&self) -> Option<&T> {
        None
    }
}

/// Typed-zero primitive: a 0-input, 1-output op that produces a value of the carried [`ArrayType`].
///
/// [`ZeroOperation`] is emitted by the linear-program transpose pass at the pullback boundary for
/// primal inputs that have no cotangent contribution accumulated onto them. Closed carriers
/// implement [`SupportsZero`] to construct the carrier-specific representation, and the carrier's
/// own trait impls then delegate to this op for [`Operation`], [`InterpretableOperation`], and
/// [`LinearOperation`] semantics.
#[derive(Clone)]
pub struct ZeroOperation {
    /// Type of the value produced when this op is interpreted.
    output_type: ArrayType,
}

impl ZeroOperation {
    /// Creates a zero op that produces values of `output_type`.
    #[inline]
    pub fn new(output_type: ArrayType) -> Self {
        Self { output_type }
    }

    /// Returns the type produced by this zero op.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }
}

impl Debug for ZeroOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Zero({})", self.output_type)
    }
}

impl Display for ZeroOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "zero")
    }
}

impl Operation<ArrayType> for ZeroOperation {
    fn name(&self) -> &'static str {
        "zero"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
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

impl<V: Typed<ArrayType> + Zero<ArrayType>> InterpretableOperation<ArrayType, V> for ZeroOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if !inputs.is_empty() {
            return Err(TracingError::InvalidInputCount { expected: 0, got: inputs.len() });
        }
        Ok(vec![V::zero(&self.output_type)?])
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for ZeroOperation {
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, ArrayType, V, LinearPrimitiveOperation<V>>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        Ok(Vec::new())
    }
}

macro_rules! impl_scalar_value_traits {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Value<ArrayType> for $ty {}

        impl Traceable<ArrayType> for $ty {}

        impl ZeroLike for $ty {
            #[inline]
            fn zero_like(&self) -> Self {
                $zero
            }
        }

        impl OneLike for $ty {
            #[inline]
            fn one_like(&self) -> Self {
                $one
            }
        }

        impl Zero<ArrayType> for $ty {
            #[inline]
            fn zero(_type: &ArrayType) -> Result<Self, TracingError> {
                Ok($zero)
            }
        }
    };
}

impl_scalar_value_traits!(bool, false, true);
impl_scalar_value_traits!(i8, 0i8, 1i8);
impl_scalar_value_traits!(i16, 0i16, 1i16);
impl_scalar_value_traits!(i32, 0i32, 1i32);
impl_scalar_value_traits!(i64, 0i64, 1i64);
impl_scalar_value_traits!(u8, 0u8, 1u8);
impl_scalar_value_traits!(u16, 0u16, 1u16);
impl_scalar_value_traits!(u32, 0u32, 1u32);
impl_scalar_value_traits!(u64, 0u64, 1u64);
impl_scalar_value_traits!(bf16, bf16::ZERO, bf16::ONE);
impl_scalar_value_traits!(f16, f16::ZERO, f16::ONE);
impl_scalar_value_traits!(f32, 0.0f32, 1.0f32);
impl_scalar_value_traits!(f64, 0.0f64, 1.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use crate::{
        tracing::Value,
        tracing_v2::{Cos, Sin, test_support},
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::{OneLike, ZeroLike};

    fn assert_scalar_value_type<V: Value<ArrayType>>(value: V, expected_type: DataType) {
        assert_eq!(value.r#type().into_owned(), ArrayType::scalar(expected_type));
    }

    fn assert_scalar_identities<V>(value: V, zero: V, one: V)
    where
        V: Value<ArrayType> + ZeroLike + OneLike + std::fmt::Debug + PartialEq,
    {
        assert_eq!(value.zero_like(), zero);
        assert_eq!(value.one_like(), one);
    }

    #[test]
    fn test_scalar_leaf_traits_report_expected_values() {
        assert_scalar_value_type(false, DataType::Boolean);
        assert_scalar_value_type(1i8, DataType::I8);
        assert_scalar_value_type(1i16, DataType::I16);
        assert_scalar_value_type(1i32, DataType::I32);
        assert_scalar_value_type(1i64, DataType::I64);
        assert_scalar_value_type(1u8, DataType::U8);
        assert_scalar_value_type(1u16, DataType::U16);
        assert_scalar_value_type(1u32, DataType::U32);
        assert_scalar_value_type(1u64, DataType::U64);
        assert_scalar_value_type(bf16::from_f32(1.25), DataType::BF16);
        assert_scalar_value_type(f16::from_f32(1.25), DataType::F16);
        assert_eq!(<f32 as Typed<ArrayType>>::r#type(&1.25f32).into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::r#type(&2.5f64).into_owned(), ArrayType::scalar(DataType::F64));
        assert_scalar_identities(false, false, true);
        assert_scalar_identities(5i32, 0i32, 1i32);
        assert_scalar_identities(5u32, 0u32, 1u32);
        assert_scalar_identities(bf16::from_f32(5.0), bf16::from_f32(0.0), bf16::from_f32(1.0));
        assert_scalar_identities(f16::from_f32(5.0), f16::from_f32(0.0), f16::from_f32(1.0));
        assert_scalar_identities(3.0f32, 0.0f32, 1.0f32);
        assert_scalar_identities(7.0f64, 0.0f64, 1.0f64);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
