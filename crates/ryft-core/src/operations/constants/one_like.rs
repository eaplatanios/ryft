use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::ProgramError;
use crate::tracing::Tracer;
use crate::types::{Type, TypeError, Typed};

/// Canonical operation name for [`OneLikeOperation`].
pub const ONE_LIKE_OPERATION_NAME: &'static str = "one_like";

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _one_ value
/// with the same [`Type`] as that input.
#[derive(Copy, Clone, Debug, Default)]
pub struct OneLikeOperation;

impl Display for OneLikeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ONE_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation<T> for OneLikeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        ONE_LIKE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Type, V: Typed<T> + OneLike> InterpretableOperation<T, V> for OneLikeOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].one_like()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`OneLikeOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`OneLikeOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsOneLike<T: Type> {
    /// Constructs an instance of [`OneLikeOperation`] for this [`Operation`] type.
    fn one_like_operation() -> Self;
}

impl<C: StagingContext<Operation: SupportsOneLike<C::Type>>> OneLike for Tracer<C> {
    #[inline]
    fn one_like(&self) -> Self {
        self.clone().unary(C::Operation::one_like_operation())
    }
}

/// Synthesizes a _one_ value from an exemplar. [`OneLike`] is the value-driven counterpart to [`One`](super::One).
/// It is what [`OneLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait OneLike {
    /// Returns a _one_ value with the same structure as `self`.
    fn one_like(&self) -> Self;
}

macro_rules! impl_one_like_for_scalar {
    ($ty:ty, $one:expr) => {
        impl OneLike for $ty {
            #[inline]
            fn one_like(&self) -> Self {
                $one
            }
        }
    };
}

impl_one_like_for_scalar!(bool, true);
impl_one_like_for_scalar!(i8, 1i8);
impl_one_like_for_scalar!(i16, 1i16);
impl_one_like_for_scalar!(i32, 1i32);
impl_one_like_for_scalar!(i64, 1i64);
impl_one_like_for_scalar!(u8, 1u8);
impl_one_like_for_scalar!(u16, 1u16);
impl_one_like_for_scalar!(u32, 1u32);
impl_one_like_for_scalar!(u64, 1u64);
impl_one_like_for_scalar!(bf16, bf16::ONE);
impl_one_like_for_scalar!(f16, f16::ONE);
impl_one_like_for_scalar!(f32, 1.0f32);
impl_one_like_for_scalar!(f64, 1.0f64);

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
    fn test_one_like() {
        assert_eq!(false.one_like(), true);
        assert_eq!(5i32.one_like(), 1i32);
        assert_eq!(5u32.one_like(), 1u32);
        assert_eq!(bf16::from_f32(5.0).one_like(), bf16::ONE);
        assert_eq!(f16::from_f32(5.0).one_like(), f16::ONE);
        assert_eq!(3.0f32.one_like(), 1.0f32);
        assert_eq!(7.0f64.one_like(), 1.0f64);

        let operation = OneLikeOperation;
        assert_eq!(Operation::<DataType>::name(&operation), ONE_LIKE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "OneLikeOperation");
        assert_eq!(format!("{operation}"), ONE_LIKE_OPERATION_NAME);
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
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
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
