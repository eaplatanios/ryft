use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::types::{DataType, Type, TypeError};

/// Canonical operation name for [`OneOperation`].
pub const ONE_OPERATION_NAME: &'static str = "one";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with ones.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a new [`OneOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }
}

impl<T: Type> Display for OneOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation<T> for OneOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        ONE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ONE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, V: Value<T> + One<T>> InterpretableOperation<T, V> for OneOperation<T> {
    #[inline]
    fn interpret(
        &self,
        _context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![V::one(&self.r#type)?])
    }
}

/// Synthesizes a _one_ value for a given [`Type`]. [`One`] is the [`Type`]-driven counterpart to
/// [`OneLike`](super::OneLike). It is what [`OneOperation`] needs for its [`InterpretableOperation`] implementation.
pub trait One<T: Type>: Sized {
    /// Returns a _one_ value for the provided [`Type`].
    fn one(r#type: &T) -> Result<Self, ProgramError>;
}

macro_rules! impl_one_for_scalar {
    ($ty:ty, $data_type:path, $one:expr) => {
        impl One<DataType> for $ty {
            #[inline]
            fn one(r#type: &DataType) -> Result<Self, ProgramError> {
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

impl_one_for_scalar!(bool, DataType::Boolean, true);
impl_one_for_scalar!(i8, DataType::I8, 1i8);
impl_one_for_scalar!(i16, DataType::I16, 1i16);
impl_one_for_scalar!(i32, DataType::I32, 1i32);
impl_one_for_scalar!(i64, DataType::I64, 1i64);
impl_one_for_scalar!(u8, DataType::U8, 1u8);
impl_one_for_scalar!(u16, DataType::U16, 1u16);
impl_one_for_scalar!(u32, DataType::U32, 1u32);
impl_one_for_scalar!(u64, DataType::U64, 1u64);
impl_one_for_scalar!(bf16, DataType::BF16, bf16::ONE);
impl_one_for_scalar!(f16, DataType::F16, f16::ONE);
impl_one_for_scalar!(f32, DataType::F32, 1.0f32);
impl_one_for_scalar!(f64, DataType::F64, 1.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
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
        assert_eq!(Operation::<DataType>::name(&operation), ONE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "OneOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "one [type=f64]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[]), Ok(vec![1.0]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[2.5]),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&OneOperation::new(DataType::F32), &mut (), &[]),
            Err(ProgramError::Type(TypeError {
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
