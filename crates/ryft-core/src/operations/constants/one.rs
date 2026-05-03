use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, TracingError};
use crate::types::{Type, TypeError, Typed};

/// Synthesizes a _one_ value for a given [`Type`]. [`One`] is the [`Type`]-driven counterpart to
/// [`OneLike`](crate::operations::constants::OneLike); it is what [`OneOperation`] needs for its
/// [`InterpretableOperation`] implementation.
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

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{DataType, TypeError};

    use super::*;

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
}
