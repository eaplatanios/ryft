use std::fmt::{Debug, Display};

use half::{bf16, f16};

use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, OperationFormatter, Traceable, TracingError, Value};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::{InterpretableOperation, LinearOperation, Operation, TranspositionContext};

/// Synthesizes a typed zero value without an exemplar.
///
/// [`Zero`] is the type-driven counterpart to [`ZeroLike`]: it is what [`ZeroOperation`] needs in
/// order to evaluate at interpretation time, since the op carries only the output type and has no
/// input values to derive a shape from. Concrete leaf value types implement this trait directly.
///
/// Wrapper types that fundamentally cannot synthesize a zero from metadata alone should use
/// exemplar-backed [`ZeroLike`] where possible. Programs containing `Zero` ops over those value
/// types must materialize them away before being interpreted.
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

/// Typed-zero primitive: a 0-input, 1-output op that produces a value of the carried type metadata.
///
/// [`ZeroOperation`] is emitted by the linear-program transpose pass at the pullback boundary for
/// primal inputs that have no cotangent contribution accumulated onto them. Closed carriers
/// implement [`SupportsZero`] to construct the carrier-specific representation, and the carrier's
/// own trait impls then delegate to this op for [`Operation`], [`InterpretableOperation`], and
/// [`LinearOperation`] semantics.
#[derive(Clone)]
pub struct ZeroOperation<T: Type = ArrayType> {
    /// Type of the value produced when this op is interpreted.
    output_type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a zero op that produces values of `output_type`.
    #[inline]
    pub fn new(output_type: T) -> Self {
        Self { output_type }
    }

    /// Returns the type produced by this zero op.
    #[inline]
    pub fn output_type(&self) -> &T {
        &self.output_type
    }
}

impl<T: Type> Debug for ZeroOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Zero({})", self.output_type)
    }
}

impl<T: Type> Display for ZeroOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "zero")
    }
}

impl<T: Type> Operation<T> for ZeroOperation<T> {
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

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for ZeroOperation<E::Type>
where
    E: DifferentiableEngine + ?Sized,
    ZeroOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        if !inputs.is_empty() {
            return Err(TracingError::InvalidInputCount { expected: 0, got: inputs.len() });
        }
        let tangent = context
            .apply_operation(
                &[],
                <E::LinearOperation as SupportsZero<E::Type, E::Value>>::zero_operation(self.output_type.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("zero jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: engine.zero(&self.output_type)?, tangent }])
    }
}

/// Synthesizes a typed unit cotangent seed without an exemplar.
///
/// [`One`] is the seed counterpart to [`Zero`]. It is intentionally fallible because not every
/// abstract descriptor admits the unit seed required by scalar-output reverse-mode transforms. For
/// example, the built-in [`ArrayType`] implementations reject non-rank-0 descriptors so `grad`
/// keeps its scalar-output semantics even though the check depends on runtime metadata.
pub trait One<T: Type>: Sized {
    /// Returns the unit cotangent seed described by `r#type`.
    fn one(r#type: &T) -> Result<Self, TracingError>;
}

/// Hidden carrier capability for staging the typed-one primitive.
///
/// `SupportsOne` is the multiplicative-identity counterpart to [`SupportsZero`]. It allows generic
/// transforms and value-level APIs to construct a carrier-specific [`OneOperation`] without knowing
/// the concrete operation enum.
#[doc(hidden)]
pub trait SupportsOne<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the typed-one primitive.
    fn one_operation(r#type: T) -> Self;

    /// Returns the type carried by a one op, or `None` for any other op variant.
    fn as_one(&self) -> Option<&T> {
        None
    }
}

/// Typed-one primitive: a 0-input, 1-output op that produces a value of the carried type metadata.
///
/// [`OneOperation`] is the staged form of [`One::one`]. It is used for unit cotangent seeds and any
/// other type-driven multiplicative identity where no exemplar value is available.
#[derive(Clone)]
pub struct OneOperation<T: Type = ArrayType> {
    /// Type of the value produced when this op is interpreted.
    output_type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a one op that produces values of `output_type`.
    #[inline]
    pub fn new(output_type: T) -> Self {
        Self { output_type }
    }

    /// Returns the type produced by this one op.
    #[inline]
    pub fn output_type(&self) -> &T {
        &self.output_type
    }
}

impl<T: Type> Debug for OneOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "One({})", self.output_type)
    }
}

impl<T: Type> Display for OneOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "one")
    }
}

impl<T: Type> Operation<T> for OneOperation<T> {
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

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for OneOperation<E::Type>
where
    E: DifferentiableEngine + ?Sized,
    OneOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        if !inputs.is_empty() {
            return Err(TracingError::InvalidInputCount { expected: 0, got: inputs.len() });
        }
        let tangent = context
            .apply_operation(
                &[],
                <E::LinearOperation as SupportsZero<E::Type, E::Value>>::zero_operation(self.output_type.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("one jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: engine.one(&self.output_type)?, tangent }])
    }
}

/// Returns a zero value with the same structure as an existing value.
///
/// [`ZeroLike`] is the local, value-level counterpart to
/// [`Engine::zero`](crate::tracing::engines::Engine::zero). When a transform already has an
/// exemplar in hand, it uses this trait instead of going back through abstract metadata. That is
/// especially important for wrappers like [`Tracer`](crate::tracing::engines::Tracer) and
/// [`JvpTracer`](crate::tracing_v2::JvpTracer), which can stage or derive a zero from their existing
/// state even when abstract synthesis alone would be insufficient. This module also ships the
/// built-in scalar implementations of [`Traceable`](crate::tracing::Traceable),
/// [`Value`](crate::tracing::Value), [`ZeroLike`], and [`OneLike`].
pub trait ZeroLike {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Hidden carrier capability for staging the exemplar-derived zero primitive.
///
/// `SupportsZeroLike` lets value-level helpers such as [`ZeroLike::zero_like`] stage a
/// [`ZeroLikeOperation`] in the backend-owned carrier without coupling that helper to a concrete
/// operation enum.
#[doc(hidden)]
pub trait SupportsZeroLike<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the zero-like primitive.
    fn zero_like_operation() -> Self;
}

impl<'engine, E> ZeroLike for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::Operation: SupportsZeroLike<E::Type, E::Value>,
{
    #[inline]
    fn zero_like(&self) -> Self {
        self.clone().unary(E::Operation::zero_like_operation())
    }
}

/// Exemplar-derived zero primitive.
///
/// [`ZeroLikeOperation`] is the staged form of [`ZeroLike::zero_like`]. It takes one exemplar input
/// and produces one output with the same abstract type, leaving concrete interpretation to the
/// value type's [`ZeroLike`] implementation.
#[derive(Copy, Clone, Default)]
pub struct ZeroLikeOperation;

impl Debug for ZeroLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ZeroLike")
    }
}

impl Display for ZeroLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "zero_like")
    }
}

impl<T: Type> Operation<T> for ZeroLikeOperation {
    fn name(&self) -> &'static str {
        "zero_like"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError {
                message: format!("zero_like expected 1 input type but got {}", input_types.len()),
            });
        }
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + ZeroLike> InterpretableOperation<T, V> for ZeroLikeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        Ok(vec![inputs[0].zero_like()])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for ZeroLikeOperation
where
    E: DifferentiableEngine + ?Sized,
    ZeroLikeOperation: Operation<E::Type>,
    E::Value: ZeroLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("zero_like jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: inputs[0].primal.zero_like(), tangent }])
    }
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

/// Hidden carrier capability for staging the exemplar-derived one primitive.
///
/// `SupportsOneLike` is the multiplicative-identity counterpart to [`SupportsZeroLike`].
#[doc(hidden)]
pub trait SupportsOneLike<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the one-like primitive.
    fn one_like_operation() -> Self;
}

impl<'engine, E> OneLike for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::Operation: SupportsOneLike<E::Type, E::Value>,
{
    #[inline]
    fn one_like(&self) -> Self {
        self.clone().unary(E::Operation::one_like_operation())
    }
}

/// Exemplar-derived one primitive.
///
/// [`OneLikeOperation`] is the staged form of [`OneLike::one_like`]. It takes one exemplar input
/// and produces one output with the same abstract type, leaving concrete interpretation to the
/// value type's [`OneLike`] implementation.
#[derive(Copy, Clone, Default)]
pub struct OneLikeOperation;

impl Debug for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "OneLike")
    }
}

impl Display for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "one_like")
    }
}

impl<T: Type> Operation<T> for OneLikeOperation {
    fn name(&self) -> &'static str {
        "one_like"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError { message: format!("one_like expected 1 input type but got {}", input_types.len()) });
        }
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + OneLike> InterpretableOperation<T, V> for OneLikeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        Ok(vec![inputs[0].one_like()])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<'_, T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        if output_cotangents.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
        }
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for OneLikeOperation
where
    E: DifferentiableEngine + ?Sized,
    OneLikeOperation: Operation<E::Type>,
    E::Value: OneLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        if inputs.len() != 1 {
            return Err(TracingError::InvalidInputCount { expected: 1, got: inputs.len() });
        }
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("one_like jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: inputs[0].primal.one_like(), tangent }])
    }
}

fn ensure_scalar_array_seed_type(r#type: &ArrayType) -> Result<(), TracingError> {
    if r#type.rank() != 0 {
        return Err(
            crate::tracing_v2::DifferentiationError::NonScalarGradientOutput { output_type: r#type.clone() }.into()
        );
    }
    Ok(())
}

fn ensure_scalar_data_type(r#type: DataType, expected: DataType) -> Result<(), TracingError> {
    if r#type != expected {
        return Err(TypeError {
            message: format!("scalar value expected data type {expected} but got {type_}", type_ = r#type),
        }
        .into());
    }
    Ok(())
}

macro_rules! impl_scalar_value_traits {
    ($ty:ty, $data_type:path, $zero:expr, $one:expr) => {
        impl Value<DataType> for $ty {}

        impl Value<ArrayType> for $ty {}

        impl Traceable<DataType> for $ty {}

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

        impl Zero<DataType> for $ty {
            #[inline]
            fn zero(r#type: &DataType) -> Result<Self, TracingError> {
                ensure_scalar_data_type(*r#type, $data_type)?;
                Ok($zero)
            }
        }

        impl Zero<ArrayType> for $ty {
            #[inline]
            fn zero(_type: &ArrayType) -> Result<Self, TracingError> {
                Ok($zero)
            }
        }

        impl One<DataType> for $ty {
            #[inline]
            fn one(r#type: &DataType) -> Result<Self, TracingError> {
                ensure_scalar_data_type(*r#type, $data_type)?;
                Ok($one)
            }
        }

        impl One<ArrayType> for $ty {
            #[inline]
            fn one(r#type: &ArrayType) -> Result<Self, TracingError> {
                ensure_scalar_array_seed_type(r#type)?;
                Ok($one)
            }
        }

        impl crate::tracing_v2::forward::Differentiable<DataType> for $ty {
            type Tangent = Self;
        }

        impl crate::tracing_v2::forward::Differentiable<ArrayType> for $ty {
            type Tangent = Self;
        }
    };
}

impl_scalar_value_traits!(bool, DataType::Boolean, false, true);
impl_scalar_value_traits!(i8, DataType::I8, 0i8, 1i8);
impl_scalar_value_traits!(i16, DataType::I16, 0i16, 1i16);
impl_scalar_value_traits!(i32, DataType::I32, 0i32, 1i32);
impl_scalar_value_traits!(i64, DataType::I64, 0i64, 1i64);
impl_scalar_value_traits!(u8, DataType::U8, 0u8, 1u8);
impl_scalar_value_traits!(u16, DataType::U16, 0u16, 1u16);
impl_scalar_value_traits!(u32, DataType::U32, 0u32, 1u32);
impl_scalar_value_traits!(u64, DataType::U64, 0u64, 1u64);
impl_scalar_value_traits!(bf16, DataType::BF16, bf16::ZERO, bf16::ONE);
impl_scalar_value_traits!(f16, DataType::F16, f16::ZERO, f16::ONE);
impl_scalar_value_traits!(f32, DataType::F32, 0.0f32, 1.0f32);
impl_scalar_value_traits!(f64, DataType::F64, 0.0f64, 1.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use crate::tracing::{InterpretableOperation, Operation, TracingError, Value};
    use crate::tracing_v2::{Cos, Sin, test_support};
    use crate::types::{ArrayType, DataType, TypeError, Typed};

    use super::{OneLike, OneLikeOperation, OneOperation, ZeroLike, ZeroLikeOperation, ZeroOperation};

    fn assert_scalar_value_type<V: Value<ArrayType>>(value: V, expected_type: DataType) {
        assert_eq!(value.r#type().into_owned(), ArrayType::scalar(expected_type));
    }

    fn assert_scalar_data_type<V: Value<DataType>>(value: V, expected_type: DataType) {
        assert_eq!(value.r#type().into_owned(), expected_type);
    }

    fn assert_scalar_identities<V>(value: V, zero: V, one: V)
    where
        V: Value<ArrayType> + ZeroLike + OneLike + std::fmt::Debug + PartialEq,
    {
        assert_eq!(value.zero_like(), zero);
        assert_eq!(value.one_like(), one);
    }

    #[test]
    fn test_zero_operation() {
        let operation = ZeroOperation::new(DataType::F64);

        assert_eq!(Operation::<DataType>::name(&operation), "zero");
        assert_eq!(format!("{operation:?}"), "Zero(f64)");
        assert_eq!(format!("{operation}"), "zero");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[]), Ok(vec![0.0]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "zero expected 0 input types but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]),
            Err(TracingError::InvalidInputCount { expected: 0, got: 1 }),
        );
    }

    #[test]
    fn test_one_operation() {
        let operation = OneOperation::new(DataType::F64);

        assert_eq!(Operation::<DataType>::name(&operation), "one");
        assert_eq!(format!("{operation:?}"), "One(f64)");
        assert_eq!(format!("{operation}"), "one");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[]), Ok(vec![1.0]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "one expected 0 input types but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]),
            Err(TracingError::InvalidInputCount { expected: 0, got: 1 }),
        );
    }

    #[test]
    fn test_zero_like_operation() {
        let operation = ZeroLikeOperation;

        assert_eq!(Operation::<DataType>::name(&operation), "zero_like");
        assert_eq!(format!("{operation:?}"), "ZeroLike");
        assert_eq!(format!("{operation}"), "zero_like");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]), Ok(vec![0.0]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );

        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "zero_like expected 1 input type but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
    }

    #[test]
    fn test_one_like_operation() {
        let operation = OneLikeOperation;

        assert_eq!(Operation::<DataType>::name(&operation), "one_like");
        assert_eq!(format!("{operation:?}"), "OneLike");
        assert_eq!(format!("{operation}"), "one_like");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]), Ok(vec![1.0]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );

        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "one_like expected 1 input type but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
    }

    #[test]
    fn test_scalar_leaf_traits_report_expected_values() {
        assert_scalar_data_type(false, DataType::Boolean);
        assert_scalar_data_type(1i8, DataType::I8);
        assert_scalar_data_type(1i16, DataType::I16);
        assert_scalar_data_type(1i32, DataType::I32);
        assert_scalar_data_type(1i64, DataType::I64);
        assert_scalar_data_type(1u8, DataType::U8);
        assert_scalar_data_type(1u16, DataType::U16);
        assert_scalar_data_type(1u32, DataType::U32);
        assert_scalar_data_type(1u64, DataType::U64);
        assert_scalar_data_type(bf16::from_f32(1.25), DataType::BF16);
        assert_scalar_data_type(f16::from_f32(1.25), DataType::F16);
        assert_eq!(<f32 as Typed<DataType>>::r#type(&1.25f32).into_owned(), DataType::F32);
        assert_eq!(<f64 as Typed<DataType>>::r#type(&2.5f64).into_owned(), DataType::F64);

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
