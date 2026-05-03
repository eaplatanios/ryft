use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, Tracer, TracingEngine, TracingError};
use crate::types::{DataType, Type, TypeError, Typed};

/// Synthesizes a _zero_ value for a given [`Type`]. [`Zero`] is the [`Type`]-driven counterpart to [`ZeroLike`]; it is
/// what [`ZeroOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait Zero<T: Type>: Sized {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(r#type: &T) -> Result<Self, TracingError>;
}

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type> {
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
        check_count!("input", input_types, 0, TypeError);
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
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![V::zero(&self.r#type)?])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`ZeroOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`ZeroOperation`] without
/// knowing which carrier is in use.
pub trait SupportsZero<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`ZeroOperation`].
    fn zero_operation(r#type: T) -> Self;

    /// Returns the [`ZeroOperation`] that this operation carrier holds, or `None` if it does not hold one.
    /// Higher-order transformation passes (e.g., the traced reverse-mode pipeline that has to materialize
    /// [`ZeroOperation`] instances into outer-trace constants before its pullback can be interpreted) use this hook
    /// to identify [`ZeroOperation`] instances without pattern-matching on concrete operation carrier types.
    fn as_zero_operation(&self) -> Option<&ZeroOperation<T>> {
        None
    }
}

/// Synthesizes a _zero_ value from an exemplar. [`ZeroLike`] is the value-driven counterpart to [`Zero`]; it is what
/// [`ZeroLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait ZeroLike {
    /// Returns a _zero_ value with the same structure as `self`.
    fn zero_like(&self) -> Self;
}

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _zero_ value
/// with the same [`Type`] as that input.
#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroLikeOperation;

impl Display for ZeroLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<DataType>>::name(self))
    }
}

impl<T: Type> Operation<T> for ZeroLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "zero_like"
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + ZeroLike> InterpretableOperation<T, V> for ZeroLikeOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].zero_like()])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`ZeroLikeOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`ZeroLikeOperation`]
/// without knowing which carrier is in use.
pub trait SupportsZeroLike<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`ZeroLikeOperation`].
    fn zero_like_operation() -> Self;
}

impl<'engine, E: TracingEngine<OperationCarrier: SupportsZeroLike<E::Type, E::Value>> + ?Sized> ZeroLike
    for Tracer<'engine, E>
{
    #[inline]
    fn zero_like(&self) -> Self {
        self.clone().unary(E::OperationCarrier::zero_like_operation())
    }
}

/// Synthesizes a _one_ value for a given [`Type`]. [`One`] is the [`Type`]-driven counterpart to
/// [`OneLike`]; it is what [`OneOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait One<T: Type>: Sized {
    /// Returns a _one_ value for the provided [`Type`].
    fn one(r#type: &T) -> Result<Self, TracingError>;
}

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with ones.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type> {
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
        check_count!("input", input_types, 0, TypeError);
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
        check_count!("input", inputs, 0, TracingError);
        Ok(vec![V::one(&self.r#type)?])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`OneOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`OneOperation`] without
/// knowing which carrier is in use.
pub trait SupportsOne<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`OneOperation`].
    fn one_operation(r#type: T) -> Self;
}

/// Synthesizes a _one_ value from an exemplar. [`OneLike`] is the value-driven counterpart to [`One`]; it is what
/// [`OneLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait OneLike {
    /// Returns a _one_ value with the same structure as `self`.
    fn one_like(&self) -> Self;
}

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _one_ value
/// with the same [`Type`] as that input.
#[derive(Copy, Clone, Debug, Default)]
pub struct OneLikeOperation;

impl Display for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<DataType>>::name(self))
    }
}

impl<T: Type> Operation<T> for OneLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "one_like"
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + OneLike> InterpretableOperation<T, V> for OneLikeOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].one_like()])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`OneLikeOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`OneLikeOperation`]
/// without knowing which carrier is in use.
pub trait SupportsOneLike<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of [`OneLikeOperation`].
    fn one_like_operation() -> Self;
}

impl<'engine, E: TracingEngine<OperationCarrier: SupportsOneLike<E::Type, E::Value>> + ?Sized> OneLike
    for Tracer<'engine, E>
{
    #[inline]
    fn one_like(&self) -> Self {
        self.clone().unary(E::OperationCarrier::one_like_operation())
    }
}

macro_rules! impl_constants_for_scalar {
    ($ty:ty, $data_type:path, $zero:expr, $one:expr) => {
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
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {} but got {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($zero)
            }
        }

        impl One<DataType> for $ty {
            #[inline]
            fn one(r#type: &DataType) -> Result<Self, TracingError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {} but got {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($one)
            }
        }
    };
}

impl_constants_for_scalar!(bool, DataType::Boolean, false, true);
impl_constants_for_scalar!(i8, DataType::I8, 0i8, 1i8);
impl_constants_for_scalar!(i16, DataType::I16, 0i16, 1i16);
impl_constants_for_scalar!(i32, DataType::I32, 0i32, 1i32);
impl_constants_for_scalar!(i64, DataType::I64, 0i64, 1i64);
impl_constants_for_scalar!(u8, DataType::U8, 0u8, 1u8);
impl_constants_for_scalar!(u16, DataType::U16, 0u16, 1u16);
impl_constants_for_scalar!(u32, DataType::U32, 0u32, 1u32);
impl_constants_for_scalar!(u64, DataType::U64, 0u64, 1u64);
impl_constants_for_scalar!(bf16, DataType::BF16, bf16::ZERO, bf16::ONE);
impl_constants_for_scalar!(f16, DataType::F16, f16::ZERO, f16::ONE);
impl_constants_for_scalar!(f32, DataType::F32, 0.0f32, 1.0f32);
impl_constants_for_scalar!(f64, DataType::F64, 0.0f64, 1.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{ArrayType, DataType, TypeError};

    use super::{
        InterpretableOperation, One, OneLike, OneLikeOperation, OneOperation, Operation, Zero, ZeroLike,
        ZeroLikeOperation, ZeroOperation,
    };

    #[test]
    fn test_zero() {
        assert_eq!(bool::zero(&DataType::Boolean), Ok(false));
        assert_eq!(i8::zero(&DataType::I8), Ok(0i8));
        assert_eq!(i16::zero(&DataType::I16), Ok(0i16));
        assert_eq!(i32::zero(&DataType::I32), Ok(0i32));
        assert_eq!(i64::zero(&DataType::I64), Ok(0i64));
        assert_eq!(u8::zero(&DataType::U8), Ok(0u8));
        assert_eq!(u16::zero(&DataType::U16), Ok(0u16));
        assert_eq!(u32::zero(&DataType::U32), Ok(0u32));
        assert_eq!(u64::zero(&DataType::U64), Ok(0u64));
        assert_eq!(bf16::zero(&DataType::BF16), Ok(bf16::ZERO));
        assert_eq!(f16::zero(&DataType::F16), Ok(f16::ZERO));
        assert_eq!(f32::zero(&DataType::F32), Ok(0.0f32));
        assert_eq!(f64::zero(&DataType::F64), Ok(0.0f64));

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
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&ZeroOperation::new(DataType::F32), &[]),
            Err(TracingError::Type(TypeError {
                message: "scalar value expected data type f64 but got f32".to_string()
            })),
        );

        let mut builder = ProgramBuilder::<DataType, f64, ZeroOperation<DataType>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), f64>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = zero [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_zero_like() {
        assert_eq!(false.zero_like(), false);
        assert_eq!(5i32.zero_like(), 0i32);
        assert_eq!(5u32.zero_like(), 0u32);
        assert_eq!(bf16::from_f32(5.0).zero_like(), bf16::ZERO);
        assert_eq!(f16::from_f32(5.0).zero_like(), f16::ZERO);
        assert_eq!(3.0f32.zero_like(), 0.0f32);
        assert_eq!(7.0f64.zero_like(), 0.0f64);

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

        let mut builder = ProgramBuilder::<DataType, f64, ZeroLikeOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                    lambda %0:f64 .
                    let %1:f64 = zero_like %0
                    in (%1)
                "}
            .trim_end(),
        );
    }

    #[test]
    fn test_one() {
        assert_eq!(bool::one(&DataType::Boolean), Ok(true));
        assert_eq!(i8::one(&DataType::I8), Ok(1i8));
        assert_eq!(i16::one(&DataType::I16), Ok(1i16));
        assert_eq!(i32::one(&DataType::I32), Ok(1i32));
        assert_eq!(i64::one(&DataType::I64), Ok(1i64));
        assert_eq!(u8::one(&DataType::U8), Ok(1u8));
        assert_eq!(u16::one(&DataType::U16), Ok(1u16));
        assert_eq!(u32::one(&DataType::U32), Ok(1u32));
        assert_eq!(u64::one(&DataType::U64), Ok(1u64));
        assert_eq!(bf16::one(&DataType::BF16), Ok(bf16::ONE));
        assert_eq!(f16::one(&DataType::F16), Ok(f16::ONE));
        assert_eq!(f32::one(&DataType::F32), Ok(1.0f32));
        assert_eq!(f64::one(&DataType::F64), Ok(1.0f64));

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
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&OneOperation::new(DataType::F32), &[]),
            Err(TracingError::Type(TypeError {
                message: "scalar value expected data type f64 but got f32".to_string()
            })),
        );

        let mut builder = ProgramBuilder::<DataType, f64, OneOperation<DataType>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), f64>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = one [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_one_like() {
        assert_eq!(false.one_like(), true);
        assert_eq!(5i32.one_like(), 1i32);
        assert_eq!(5u32.one_like(), 1u32);
        assert_eq!(bf16::from_f32(5.0).one_like(), bf16::ONE);
        assert_eq!(f16::from_f32(5.0).one_like(), f16::ONE);
        assert_eq!(3.0f32.one_like(), 1.0f32);
        assert_eq!(7.0f64.one_like(), 1.0f64);

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

        let mut builder = ProgramBuilder::<DataType, f64, OneLikeOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = one_like %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
