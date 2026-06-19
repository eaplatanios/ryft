use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::types::{DataType, Type, TypeError, Typed};

/// Canonical operation name for [`FillOperation`].
pub const FILL_OPERATION_NAME: &'static str = "fill";

/// [`Operation`] that has no inputs and that produces a single output equal to the [`Type`] it holds (i.e., its
/// `r#type` field) filled with a captured scalar `V` value. [`FillOperation`] is the scalar-broadcast counterpart of
/// [`ConstantOperation`](super::ConstantOperation). Rather than carrying a fully typed value, it carries a target
/// [`Type`] plus a scalar `V` and synthesizes its output value through the [`Fill`] trait when interpreted. For arrays,
/// this corresponds to an array of the held type and shape with every element set to the captured scalar. It mirrors
/// [`ZeroOperation`](super::ZeroOperation) and [`OneOperation`](super::OneOperation), generalizing the fixed `zero` or
/// `one` value to an arbitrary captured scalar value.
#[derive(Copy, Clone, Debug)]
pub struct FillOperation<T: Type, V> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,

    /// Captured scalar value used to fill the produced value when this operation is interpreted.
    value: V,
}

impl<T: Type, V> FillOperation<T, V> {
    /// Creates a new [`FillOperation`] with the provided output type and fill value.
    #[inline]
    pub fn new(r#type: T, value: V) -> Self {
        Self { r#type, value }
    }

    /// Returns the type of the value produced by this [`FillOperation`].
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }

    /// Returns the captured scalar value used to fill the produced value for this [`FillOperation`].
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<T: Type, V: Display> Display for FillOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Display> Operation<T> for FillOperation<T, V> {
    #[inline]
    fn name(&self) -> &'static str {
        FILL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, FILL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("value", &self.value)
        })
    }
}

impl<T: Type, V: Clone + Display, W: Value<T> + Fill<T, V>> InterpretableOperation<T, W> for FillOperation<T, V> {
    #[inline]
    fn interpret(
        &self,
        _context: &mut <W as Value<T>>::InterpretationContext,
        inputs: &[W],
    ) -> Result<Vec<W>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![W::fill(&self.r#type, self.value.clone())?])
    }
}

/// Synthesizes a value for a given [`Type`] filled with a captured scalar. [`Fill`] is the [`Type`]-driven
/// counterpart needed by [`FillOperation`] for its [`InterpretableOperation`] implementation. It sits alongside
/// [`Zero`](super::Zero) and [`One`](super::One) in the same type-driven family, but generalizes the fixed `zero`
/// or `one` value to an arbitrary scalar `V` value supplied at the call site.
pub trait Fill<T: Type, V>: Sized {
    /// Returns a value of `type` with every element set to `value`.
    fn fill(r#type: &T, value: V) -> Result<Self, ProgramError>;
}

// TODO(eaplatanios): Move to `ryft_core::scalars`.
macro_rules! impl_fill_for_scalar {
    ($ty:ty) => {
        impl Fill<DataType, $ty> for $ty {
            #[inline]
            fn fill(r#type: &DataType, value: $ty) -> Result<Self, ProgramError> {
                let value_type = <$ty as Typed<DataType>>::r#type(&value).into_owned();
                if *r#type != value_type {
                    return Err(TypeError {
                        message: format!("scalar value expected data type {value_type} but got {}", r#type),
                    }
                    .into());
                }
                Ok(value)
            }
        }
    };
}

impl_fill_for_scalar!(bool);
impl_fill_for_scalar!(i8);
impl_fill_for_scalar!(i16);
impl_fill_for_scalar!(i32);
impl_fill_for_scalar!(i64);
impl_fill_for_scalar!(u8);
impl_fill_for_scalar!(u16);
impl_fill_for_scalar!(u32);
impl_fill_for_scalar!(u64);
impl_fill_for_scalar!(bf16);
impl_fill_for_scalar!(f16);
impl_fill_for_scalar!(f32);
impl_fill_for_scalar!(f64);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{DataType, TypeError};

    use super::*;

    #[test]
    fn test_fill() {
        assert_eq!(f64::fill(&DataType::F64, 3.5), Ok(3.5));
        assert_eq!(
            f64::fill(&DataType::F32, 3.5),
            Err(ProgramError::Type(TypeError {
                message: "scalar value expected data type f64 but got f32".to_string()
            })),
        );

        let operation = FillOperation::new(DataType::F64, 3.5);

        assert_eq!(Operation::<DataType>::name(&operation), FILL_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "FillOperation { type: F64, value: 3.5 }");
        assert_eq!(format!("{operation}"), "fill [type=f64, value=3.5]");
        assert_eq!(operation.r#type(), &DataType::F64);
        assert_eq!(operation.value(), &3.5);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[]), Ok(vec![DataType::F64]));
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[]), Ok(vec![3.5]));
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64]),
            Err(TypeError { message: "expected 0 inputs but got 1".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &mut (), &[0.0]),
            Err(ProgramError::InvalidInputCount { expected: 0, actual: 1 }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&FillOperation::new(DataType::F32, 3.5), &mut (), &[]),
            Err(ProgramError::Type(TypeError {
                message: "scalar value expected data type f64 but got f32".to_string()
            })),
        );

        let mut builder = ProgramBuilder::<DataType, f64, FillOperation<DataType, f64>>::new();
        let output = builder.add_instruction(operation, vec![]).unwrap()[0];
        let program = builder.build::<(), f64>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = fill [type=f64, value=3.5]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
