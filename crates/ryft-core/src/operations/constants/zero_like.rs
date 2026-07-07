use std::fmt::Display;

use crate::contexts::Context;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::types::{Type, TypeError};

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

impl ElementwiseOperation for ZeroLikeOperation {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Value + ZeroLike, C> InterpretableOperation<V, C> for ZeroLikeOperation {
    #[inline]
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].zero_like()])
    }
}

impl<C: Context<Operation: From<ZeroLikeOperation>>> PartiallyEvaluatableOperation<C> for ZeroLikeOperation {}

/// Synthesizes a _zero_ value from an exemplar. [`ZeroLike`] is the value-driven counterpart to [`Zero`](super::Zero).
/// It is what [`ZeroLikeOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait ZeroLike {
    /// Returns a _zero_ value with the same structure as `self`.
    fn zero_like(&self) -> Self;
}

impl<V: Value<DispatchDomain: Context<Operation: From<ZeroLikeOperation>>>> ZeroLike for V {
    #[inline]
    fn zero_like(&self) -> Self {
        self.dispatch_domain()
            .bind(ZeroLikeOperation, &[self.clone()])
            .expect("`zero_like` operation failed")
            .remove(0)
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::scalars::Scalar;
    use crate::types::{ArrayType, DataType, TypeError};

    use super::*;

    #[test]
    fn test_zero_like() {
        assert_eq!(Scalar::from(false).zero_like(), Scalar::from(false));
        assert_eq!(Scalar::from(5i32).zero_like(), Scalar::from(0i32));
        assert_eq!(Scalar::from(5u32).zero_like(), Scalar::from(0u32));
        assert_eq!(Scalar::from(bf16::from_f32(5.0)).zero_like(), Scalar::from(bf16::ZERO));
        assert_eq!(Scalar::from(f16::from_f32(5.0)).zero_like(), Scalar::from(f16::ZERO));
        assert_eq!(Scalar::from(3.0f32).zero_like(), Scalar::from(0.0f32));
        assert_eq!(Scalar::from(7.0f64).zero_like(), Scalar::from(0.0f64));

        let operation = ZeroLikeOperation;
        assert_eq!(Operation::<DataType>::name(&operation), ZERO_LIKE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ZeroLikeOperation");
        assert_eq!(format!("{operation}"), ZERO_LIKE_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<Scalar, crate::EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[Scalar::from(2.5)],
            ),
            Ok(vec![Scalar::from(0.0)]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[ArrayType::scalar(DataType::F32)]),
            Ok(vec![ArrayType::scalar(DataType::F32)]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<Scalar, crate::EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        let mut builder = ProgramBuilder::<Scalar, ZeroLikeOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
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
