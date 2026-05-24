use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use half::{bf16, f16};
use crate::{AddOperation, ZeroOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::types::{Type, TypeError, Typed};

/// Canonical operation name for [`ConstantLikeOperation`].
pub const CONSTANT_LIKE_OPERATION_NAME: &'static str = "constant_like";

/// [`Operation`] that takes one exemplar input and produces a single output that has the same [`Type`] as the input,
/// filled with the captured `F` constant. Mirrors [`ZeroLikeOperation`](crate::ZeroLikeOperation) and
/// [`OneLikeOperation`](crate::OneLikeOperation) but parameterized on the captured constant type `F` so that transform
/// rules can stage arbitrary scalar constants without requiring the value type to implement `From<F>` directly.
#[derive(Copy, Clone, Debug)]
pub struct ConstantLikeOperation<T: Type, F> {
    /// Captured constant value produced by this [`Operation`] when interpreted against an exemplar.
    value: F,

    /// [`PhantomData`] marker tying the captured constant to the [`Type`] it is interpreted against.
    marker: PhantomData<T>,
}

impl<T: Type, F> ConstantLikeOperation<T, F> {
    /// Creates a new [`ConstantLikeOperation`] capturing the provided constant value.
    #[inline]
    pub fn new(value: F) -> Self {
        Self { value, marker: PhantomData }
    }

    /// Returns the captured constant value produced by this operation.
    #[inline]
    pub fn value(&self) -> &F {
        &self.value
    }
}

impl<T: Type, F: Display> Display for ConstantLikeOperation<T, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(CONSTANT_LIKE_OPERATION_NAME)
    }
}

impl<T: Type, F: Debug + Display> Operation<T> for ConstantLikeOperation<T, F> {
    #[inline]
    fn name(&self) -> &'static str {
        CONSTANT_LIKE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONSTANT_LIKE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl<T: Type, F: Clone + Debug + Display, V: Typed<T> + ConstantLike<F>> InterpretableOperation<T, V>
    for ConstantLikeOperation<T, F>
{
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].constant_like(self.value.clone())])
    }
}

/// Trait that represents [`Operation`] types that support/include [`ConstantLikeOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`ConstantLikeOperation`] without
/// knowing which type is in use.
pub trait SupportsConstantLike<T: Type, V: Traceable<T>, F> {
    /// Constructs an instance of [`ConstantLikeOperation`] for this [`Operation`] type.
    fn constant_like_operation(value: F) -> Self;
}

impl<C: Context<Operation: SupportsConstantLike<C::Type, C::Value, F>>, F> ConstantLike<F> for Tracer<C> {
    #[inline]
    fn constant_like(&self, value: F) -> Self {
        self.clone().unary(C::Operation::constant_like_operation(value))
    }
}

/// Synthesizes a value with the same [`Type`] as an exemplar, filled with a captured constant `F`. [`ConstantLike`] is
/// the value-driven counterpart used by [`ConstantLikeOperation`] for its [`InterpretableOperation`] implementation;
/// it sits alongside [`ZeroLike`](crate::ZeroLike) and [`OneLike`](crate::OneLike) in the same exemplar-driven family
/// but generalizes the captured constant from a fixed `zero`/`one` to an arbitrary value of type `F`.
pub trait ConstantLike<F>: Sized {
    /// Returns a value with the same [`Type`] as `self` filled with the provided `F` (i.e., `value`).
    fn constant_like(&self, value: F) -> Self;
}

macro_rules! impl_constant_like_for_scalar {
    ($ty:ty) => {
        impl ConstantLike<$ty> for $ty {
            #[inline]
            fn constant_like(&self, value: $ty) -> Self {
                value
            }
        }
    };
}

impl_constant_like_for_scalar!(i8);
impl_constant_like_for_scalar!(i16);
impl_constant_like_for_scalar!(i32);
impl_constant_like_for_scalar!(i64);
impl_constant_like_for_scalar!(u8);
impl_constant_like_for_scalar!(u16);
impl_constant_like_for_scalar!(u32);
impl_constant_like_for_scalar!(u64);
impl_constant_like_for_scalar!(bf16);
impl_constant_like_for_scalar!(f16);
impl_constant_like_for_scalar!(f32);
impl_constant_like_for_scalar!(f64);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[test]
    fn test_constant_like() {
        let operation = ConstantLikeOperation::<DataType, f64>::new(3.5);

        assert_eq!(Operation::<DataType>::name(&operation), CONSTANT_LIKE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            "ConstantLikeOperation { value: 3.5, marker: PhantomData<ryft_core::types::data_types::DataType> }"
        );
        assert_eq!(format!("{operation}"), CONSTANT_LIKE_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[0.0]), Ok(vec![3.5]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );

        let mut builder = ProgramBuilder::<DataType, f64, ConstantLikeOperation<DataType, f64>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = constant_like [value=3.5] %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
