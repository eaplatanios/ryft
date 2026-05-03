use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, TracingError};
use crate::types::{Type, TypeError, Typed};

/// Synthesizes a _zero_ value for a given [`Type`]. [`Zero`] is the [`Type`]-driven counterpart to
/// [`ZeroLike`](crate::operations::constants::ZeroLike); it is what [`ZeroOperation`] needs for its
/// [`InterpretableOperation`] implementation.
pub trait Zero<T: Type>: Sized {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(r#type: &T) -> Result<Self, TracingError>;
}

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &'static str = "zero";

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
        formatter.write_str(ZERO_OPERATION_NAME)
    }
}

impl<T: Type> Operation<T> for ZeroOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        ZERO_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ZERO_OPERATION_NAME)?
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
        assert_eq!(Operation::<DataType>::name(&operation), ZERO_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ZeroOperation { type: F64 }");
        assert_eq!(format!("{operation}"), ZERO_OPERATION_NAME);
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
}
