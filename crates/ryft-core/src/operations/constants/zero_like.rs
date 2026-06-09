use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::ProgramError;
use crate::tracing::Tracer;
use crate::types::{Type, TypeError, Typed};

/// Canonical operation name for [`ZeroLikeOperation`].
pub const ZERO_LIKE_OPERATION_NAME: &'static str = "zero_like";

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _zero_ value
/// with the same [`Type`] as that input.
#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroLikeOperation;

impl Display for ZeroLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ZERO_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation<T> for ZeroLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        ZERO_LIKE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + ZeroLike> InterpretableOperation<T, V> for ZeroLikeOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].zero_like()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`ZeroLikeOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`ZeroLikeOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsZeroLike<T: Type> {
    /// Constructs an instance of [`ZeroLikeOperation`] for this [`Operation`] type.
    fn zero_like_operation() -> Self;
}

impl<C: StagingContext<Operation: SupportsZeroLike<C::Type>>> ZeroLike for Tracer<C> {
    #[inline]
    fn zero_like(&self) -> Self {
        self.clone().unary(C::Operation::zero_like_operation())
    }
}

/// Synthesizes a _zero_ value from an exemplar. [`ZeroLike`] is the value-driven counterpart to [`Zero`](super::Zero).
/// It is what [`ZeroLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait ZeroLike {
    /// Returns a _zero_ value with the same structure as `self`.
    fn zero_like(&self) -> Self;
}

macro_rules! impl_zero_like_for_scalar {
    ($ty:ty, $zero:expr) => {
        impl ZeroLike for $ty {
            #[inline]
            fn zero_like(&self) -> Self {
                $zero
            }
        }
    };
}

impl_zero_like_for_scalar!(bool, false);
impl_zero_like_for_scalar!(i8, 0i8);
impl_zero_like_for_scalar!(i16, 0i16);
impl_zero_like_for_scalar!(i32, 0i32);
impl_zero_like_for_scalar!(i64, 0i64);
impl_zero_like_for_scalar!(u8, 0u8);
impl_zero_like_for_scalar!(u16, 0u16);
impl_zero_like_for_scalar!(u32, 0u32);
impl_zero_like_for_scalar!(u64, 0u64);
impl_zero_like_for_scalar!(bf16, bf16::ZERO);
impl_zero_like_for_scalar!(f16, f16::ZERO);
impl_zero_like_for_scalar!(f32, 0.0f32);
impl_zero_like_for_scalar!(f64, 0.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{ArrayType, DataType, TypeError};

    use super::*;

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
        assert_eq!(Operation::<DataType>::name(&operation), ZERO_LIKE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ZeroLikeOperation");
        assert_eq!(format!("{operation}"), ZERO_LIKE_OPERATION_NAME);
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
            Err(ProgramError::InvalidInputCount { expected: 1, got: 0 }),
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
}
