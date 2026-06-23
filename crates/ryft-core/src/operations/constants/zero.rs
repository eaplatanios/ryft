use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::{EagerContext, StagingContext};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{DataType, Type, TypeError};

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &'static str = "zero";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a new [`ZeroOperation`].
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

impl<T: Type> Display for ZeroOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
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

impl<T: Type, V: Value<T, InterpretationContext: Zero<T, V>>> InterpretableOperation<T, V> for ZeroOperation<T> {
    #[inline]
    fn interpret(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.zero(&self.r#type)?])
    }
}

/// Represents [`Operation`]s that may be or may be carrying [`ZeroOperation`] payloads. [`MaybeZeroOperation`] says
/// that a borrowed operation value can be inspected to determine whether it is a [`ZeroOperation`] (or a wrapper of
/// one). Structural zero analyses use this borrowed form so that they can recognize existing staged zeros without
/// cloning, moving, allocating, or manufacturing placeholder operations.
pub trait MaybeZeroOperation<T: Type> {
    /// Returns `true` if `self` is a [`ZeroOperation`] (or a wrapper of one).
    fn is_zero_operation(&self) -> bool;
}

impl<T: Type, O> MaybeZeroOperation<T> for O
where
    for<'operation> &'operation ZeroOperation<T>: TryFrom<&'operation O>,
{
    #[inline]
    fn is_zero_operation(&self) -> bool {
        <&ZeroOperation<T>>::try_from(self).is_ok()
    }
}

/// Represents the ability to synthesize a _zero_ value for a given [`Type`] in an interpretation context. [`Zero`] is
/// the [`Type`]-driven counterpart to [`ZeroLike`](super::ZeroLike). It is what [`ZeroOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait Zero<T: Type, V: Value<T>> {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(&self, r#type: &T) -> Result<V, ProgramError>;
}

impl<C: StagingContext<Operation: From<ZeroOperation<C::Type>>>> Zero<C::Type, Tracer<C>> for C {
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(ZeroOperation::new(r#type.clone()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

macro_rules! impl_zero_for_scalar {
    ($ty:ty, $data_type:path, $zero:expr) => {
        impl<O: Operation<DataType>> Zero<DataType, $ty> for EagerContext<DataType, $ty, O> {
            #[inline]
            fn zero(&self, r#type: &DataType) -> Result<$ty, ProgramError> {
                if *r#type != $data_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {} but got {}", $data_type, r#type),
                    }
                    .into());
                }
                Ok($zero)
            }
        }
    };
}

impl_zero_for_scalar!(bool, DataType::Boolean, false);
impl_zero_for_scalar!(i8, DataType::I8, 0i8);
impl_zero_for_scalar!(i16, DataType::I16, 0i16);
impl_zero_for_scalar!(i32, DataType::I32, 0i32);
impl_zero_for_scalar!(i64, DataType::I64, 0i64);
impl_zero_for_scalar!(u8, DataType::U8, 0u8);
impl_zero_for_scalar!(u16, DataType::U16, 0u16);
impl_zero_for_scalar!(u32, DataType::U32, 0u32);
impl_zero_for_scalar!(u64, DataType::U64, 0u64);
impl_zero_for_scalar!(bf16, DataType::BF16, bf16::ZERO);
impl_zero_for_scalar!(f16, DataType::F16, f16::ZERO);
impl_zero_for_scalar!(f32, DataType::F32, 0.0f32);
impl_zero_for_scalar!(f64, DataType::F64, 0.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(EagerContext::<DataType, bool, ZeroOperation<DataType>>::new().zero(&DataType::Boolean), Ok(false));
        assert_eq!(EagerContext::<DataType, i8, ZeroOperation<DataType>>::new().zero(&DataType::I8), Ok(0i8));
        assert_eq!(EagerContext::<DataType, i16, ZeroOperation<DataType>>::new().zero(&DataType::I16), Ok(0i16));
        assert_eq!(EagerContext::<DataType, i32, ZeroOperation<DataType>>::new().zero(&DataType::I32), Ok(0i32));
        assert_eq!(EagerContext::<DataType, i64, ZeroOperation<DataType>>::new().zero(&DataType::I64), Ok(0i64));
        assert_eq!(EagerContext::<DataType, u8, ZeroOperation<DataType>>::new().zero(&DataType::U8), Ok(0u8));
        assert_eq!(EagerContext::<DataType, u16, ZeroOperation<DataType>>::new().zero(&DataType::U16), Ok(0u16));
        assert_eq!(EagerContext::<DataType, u32, ZeroOperation<DataType>>::new().zero(&DataType::U32), Ok(0u32));
        assert_eq!(EagerContext::<DataType, u64, ZeroOperation<DataType>>::new().zero(&DataType::U64), Ok(0u64));
        assert_eq!(
            EagerContext::<DataType, bf16, ZeroOperation<DataType>>::new().zero(&DataType::BF16),
            Ok(bf16::ZERO),
        );
        assert_eq!(EagerContext::<DataType, f16, ZeroOperation<DataType>>::new().zero(&DataType::F16), Ok(f16::ZERO));
        assert_eq!(EagerContext::<DataType, f32, ZeroOperation<DataType>>::new().zero(&DataType::F32), Ok(0.0f32));
        assert_eq!(EagerContext::<DataType, f64, ZeroOperation<DataType>>::new().zero(&DataType::F64), Ok(0.0f64));

        let operation = ZeroOperation::new(DataType::F64);
        assert_eq!(Operation::<DataType>::name(&operation), ZERO_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ZeroOperation { type: F64 }");
        assert_eq!(format!("{operation}"), "zero [type=f64]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[]),
            Ok(vec![0.0]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[2.5]),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(
                &ZeroOperation::new(DataType::F32),
                &EagerContext::new(),
                &[],
            ),
            Err(ProgramError::Type(TypeError {
                message: "scalar value expected data type f64 but got f32".to_string(),
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
