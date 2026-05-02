use std::fmt::{Debug, Display};

use half::{bf16, f16};

use crate::macros::check_input_count;
use crate::operations::constants::{One, OneOperation, SupportsZero, Zero, ZeroOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::transposition::{LinearOperation, TranspositionContext};
use crate::tracing::{AtomId, Traceable, TracingError, Value};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for ZeroOperation<E::Type>
where
    E: LinearizableEngine + ?Sized,
    ZeroOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 0, TracingError);
        let tangent = context
            .apply_operation(
                &[],
                <E::LinearOperationCarrier as SupportsZero<E::Type, E::Value>>::zero_operation(self.r#type.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("zero jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: context.engine.zero(&self.r#type)?, tangent }])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for OneOperation<E::Type>
where
    E: LinearizableEngine + ?Sized,
    OneOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 0, TracingError);
        let tangent = context
            .apply_operation(
                &[],
                <E::LinearOperationCarrier as SupportsZero<E::Type, E::Value>>::zero_operation(self.r#type.clone()),
                1,
            )?
            .into_iter()
            .next()
            .expect("one jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: context.engine.one(&self.r#type)?, tangent }])
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

/// Trait that represents [`Operation`] carrier types that support/include [`ZeroLikeOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`ZeroLikeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsZeroLike<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the zero-like [`Operation`].
    fn zero_like_operation() -> Self;
}

impl<'engine, E> ZeroLike for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::OperationCarrier: SupportsZeroLike<E::Type, E::Value>,
{
    #[inline]
    fn zero_like(&self) -> Self {
        self.clone().unary(E::OperationCarrier::zero_like_operation())
    }
}

/// Exemplar-derived zero primitive.
///
/// [`ZeroLikeOperation`] is the staged form of [`ZeroLike::zero_like`]. It takes one exemplar input
/// and produces one output with the same abstract type, leaving concrete interpretation to the
/// value type's [`ZeroLike`] implementation.
#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroLikeOperation;

impl Display for ZeroLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl<T: Type> Operation<T> for ZeroLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "zero_like"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + ZeroLike> InterpretableOperation<T, V> for ZeroLikeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].zero_like()])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for ZeroLikeOperation
where
    E: LinearizableEngine + ?Sized,
    ZeroLikeOperation: Operation<E::Type>,
    E::Value: ZeroLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperationCarrier as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
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

/// Trait that represents [`Operation`] carrier types that support/include [`OneLikeOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`OneLikeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsOneLike<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the one-like [`Operation`].
    fn one_like_operation() -> Self;
}

impl<'engine, E> OneLike for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::OperationCarrier: SupportsOneLike<E::Type, E::Value>,
{
    #[inline]
    fn one_like(&self) -> Self {
        self.clone().unary(E::OperationCarrier::one_like_operation())
    }
}

/// Exemplar-derived one primitive.
///
/// [`OneLikeOperation`] is the staged form of [`OneLike::one_like`]. It takes one exemplar input
/// and produces one output with the same abstract type, leaving concrete interpretation to the
/// value type's [`OneLike`] implementation.
#[derive(Copy, Clone, Debug, Default)]
pub struct OneLikeOperation;

impl Display for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl<T: Type> Operation<T> for OneLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "one_like"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + OneLike> InterpretableOperation<T, V> for OneLikeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].one_like()])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for OneLikeOperation
where
    E: LinearizableEngine + ?Sized,
    OneLikeOperation: Operation<E::Type>,
    E::Value: OneLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperationCarrier as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
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

        impl crate::tracing_v2::differentiation::Differentiable<DataType> for $ty {
            type Tangent = Self;
        }

        impl crate::tracing_v2::differentiation::Differentiable<ArrayType> for $ty {
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
    use indoc::indoc;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::tracing::engines::{ScalarEngine, TracingEngine};
    use crate::tracing::{Program, TracingError, Value};
    use crate::tracing_v2::{Cos, ScalarOperation, Sin};
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
        assert_eq!(format!("{operation:?}"), "ZeroOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "zero");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[]), Ok(vec![0.0]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
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
        assert_eq!(format!("{operation:?}"), "OneOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "one");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[]), Ok(vec![1.0]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
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
        assert_eq!(format!("{operation:?}"), "ZeroLikeOperation");
        assert_eq!(format!("{operation}"), "zero_like");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]), Ok(vec![0.0]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );

        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
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
        assert_eq!(format!("{operation:?}"), "OneLikeOperation");
        assert_eq!(format!("{operation}"), "one_like");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.5]), Ok(vec![1.0]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );

        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
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

        let engine = ScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            engine.interpret_and_trace(|x| Ok(x.sin()), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());

        let engine = ScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            engine.interpret_and_trace(|x| Ok(x.sin()), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
