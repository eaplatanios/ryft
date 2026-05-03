use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::{Traceable, Tracer, TracingEngine, TracingError};
use crate::types::{DataType, Type, TypeError, Typed};

/// Synthesizes a _one_ value from an exemplar. [`OneLike`] is the value-driven counterpart to
/// [`One`](crate::operations::constants::One); it is what [`OneLikeOperation`] needs for its
/// [`InterpretableOperation`] implementation.
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

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::tracing::{ProgramBuilder, TracingError};
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
