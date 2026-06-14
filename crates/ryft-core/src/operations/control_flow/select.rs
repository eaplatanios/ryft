use std::fmt::Display;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::constants::Zero;
use crate::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`SelectOperation`].
pub const SELECT_OPERATION_NAME: &'static str = "select";

/// [`Operation`] that performs an elementwise selection between two values driven by a Boolean condition. Refer to the
/// documentation of [`Select`] for more information.
#[derive(Copy, Clone, Debug)]
pub struct SelectOperation;

impl Display for SelectOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SELECT_OPERATION_NAME)
    }
}

impl Operation<DataType> for SelectOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        if input_types[0] != DataType::Boolean {
            return Err(TypeError {
                message: format!("select condition data type {} is not {}", input_types[0], DataType::Boolean),
            });
        }
        if input_types[1] != input_types[2] {
            return Err(TypeError {
                message: format!(
                    "select on_true data type {} differs from on_false data type {}",
                    input_types[1], input_types[2],
                ),
            });
        }
        Ok(vec![input_types[1]])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME).map(|_| ())
    }
}

impl Operation<ArrayType> for SelectOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        if input_types[0].data_type() != DataType::Boolean {
            return Err(TypeError {
                message: format!(
                    "select condition data type {} is not {}",
                    input_types[0].data_type(),
                    DataType::Boolean,
                ),
            });
        }
        if input_types[0].shape() != input_types[1].shape() {
            return Err(TypeError {
                message: format!(
                    "select condition shape {} differs from on_true shape {}",
                    input_types[0].shape(),
                    input_types[1].shape(),
                ),
            });
        }
        if input_types[1].shape() != input_types[2].shape() {
            return Err(TypeError {
                message: format!(
                    "select on_true shape {} differs from on_false shape {}",
                    input_types[1].shape(),
                    input_types[2].shape(),
                ),
            });
        }
        if input_types[1].data_type() != input_types[2].data_type() {
            return Err(TypeError {
                message: format!(
                    "select on_true data type {} differs from on_false data type {}",
                    input_types[1].data_type(),
                    input_types[2].data_type(),
                ),
            });
        }
        Ok(vec![input_types[1].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME).map(|_| ())
    }
}

impl<V: Value<DataType> + BooleanLike + Select<Condition = bool>> InterpretableOperation<DataType, V>
    for SelectOperation
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![V::select(&inputs[0].boolean()?, &inputs[1], &inputs[2])?])
    }
}

impl<V: Value<ArrayType> + Select<Condition = V>> InterpretableOperation<ArrayType, V> for SelectOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![V::select(&inputs[0], &inputs[1], &inputs[2])?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`SelectOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`SelectOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsSelect<T: Type> {
    /// Constructs an instance of [`SelectOperation`] for this [`Operation`] type.
    fn select_operation() -> Self;
}

/// Represents the ability to perform an elementwise selection between two values driven by a condition. This is the
/// direct analogue of JAX's [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) in its
/// three-argument form.
///
/// For arrays, `Self::select(condition, on_true, on_false)` returns a value whose `i`-th element equals `on_true`'s
/// `i`-th element when the corresponding element of `condition` is true, and `on_false`'s otherwise. The condition and
/// branch value types may differ: for scalar values the condition is a plain [`bool`], while array value types pair
/// with a Boolean-typed condition array of the same shape. Value types that participate in closed staged operation sets
/// (e.g., [`Tracer`]) use `Condition = Self`, representing the condition as a [`DataType::Boolean`] value.
///
/// # Example
///
/// The following example shows how to use [`Select`] in practice:
///
/// ```rust
/// # use ryft_core::operations::control_flow::Select;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Scalar values use a plain `bool` condition.
/// assert_eq!(f64::select(&true, &2.0, &3.0)?, 2.0);
/// assert_eq!(f64::select(&false, &2.0, &3.0)?, 3.0);
///
/// // Array values pair with a Boolean-typed condition array of the same shape.
/// let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
/// let condition = Array::new(condition_type, vec![1.0, 0.0, 1.0]);
/// let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
/// let on_false = Array::vector(vec![4.0, 5.0, 6.0]);
/// let output = Array::select(&condition, &on_true, &on_false)?;
/// assert_eq!(output.values, vec![1.0, 5.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub trait Select: Sized {
    /// Condition value type that drives the selection.
    type Condition;

    /// Selects from `on_true` and `on_false` based on `condition`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    fn select(condition: &Self::Condition, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError>;
}

macro_rules! impl_select_for_scalar {
    ($($type:ty),* $(,)?) => {
        $(
            impl Select for $type {
                type Condition = bool;

                #[inline]
                fn select(condition: &bool, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
                    Ok(if *condition { *on_true } else { *on_false })
                }
            }
        )*
    };
}

impl_select_for_scalar!(bf16, f16, f32, f64);

impl<C: StagingContext<Operation: SupportsSelect<C::Type>>> Select for Tracer<C> {
    type Condition = Self;

    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let mut outputs = condition
            .context()
            .stage_operation(C::Operation::select_operation(), &[condition, on_true, on_false])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + Zero<ArrayType> + Select<Condition = V>> Select for Tangent<ArrayType, V> {
    type Condition = V;

    fn select(condition: &V, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let output_types = SelectOperation.infer_output_types(&[
            condition.r#type().into_owned(),
            on_true.r#type().into_owned(),
            on_false.r#type().into_owned(),
        ])?;
        check_count!("output", output_types, 1, ProgramError);
        if on_true.is_zero() && on_false.is_zero() {
            return Ok(Self::Zero(output_types.into_iter().next().unwrap()));
        }
        let materialize = |tangent: &Self| match tangent {
            Self::Zero(r#type) => V::zero(r#type),
            Self::Value(value) => Ok(value.clone()),
        };
        Ok(Self::Value(V::select(condition, &materialize(on_true)?, &materialize(on_false)?)?))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{Shape, Size};

    use super::*;

    #[test]
    fn test_select() {
        let operation = SelectOperation;

        // Operation identity.
        assert_eq!(Operation::<ArrayType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), SELECT_OPERATION_NAME);

        // Scalar (`DataType`) type inference validates the Boolean condition and matching branch data types.
        assert_eq!(
            operation.infer_output_types(&[DataType::Boolean, DataType::F64, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            operation.infer_output_types(&[DataType::F64, DataType::F64, DataType::F64]),
            Err(TypeError { message: "select condition data type f64 is not bool".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[DataType::Boolean, DataType::F64, DataType::F32]),
            Err(TypeError { message: "select on_true data type f64 differs from on_false data type f32".to_string() }),
        );

        // Scalar interpretation treats the in-band condition as true exactly when it is nonzero.
        assert_eq!(operation.interpret(&[1.0f64, 2.0f64, 3.0f64]), Ok(vec![2.0]));
        assert_eq!(operation.interpret(&[0.0f64, 2.0f64, 3.0f64]), Ok(vec![3.0]));

        // Scalar values decode and reinterpret their in-band Boolean payload through `BooleanLike`.
        assert_eq!(1.5f64.boolean(), Ok(true));
        assert_eq!(0.0f64.boolean(), Ok(false));
        assert_eq!(2.0f64.as_boolean(), 1.0);
        assert_eq!(0.0f32.as_boolean(), 0.0);
        assert_eq!(f16::from_f64(-3.0).as_boolean(), f16::ONE);
        assert_eq!(bf16::ZERO.boolean(), Ok(false));

        // Type inference validates the condition and branch types and returns the branch type.
        let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
        let branch_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            operation.infer_output_types(&[condition_type.clone(), branch_type.clone(), branch_type.clone()]),
            Ok(vec![branch_type.clone()]),
        );

        // Interpretation picks per-element between the two branches.
        let condition = TestArray::new(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let on_false = TestArray::vector(vec![4.0, 5.0, 6.0]);
        let output = operation.interpret(&[condition, on_true, on_false]).unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].values, vec![1.0, 5.0, 3.0]);

        // The scalar implementations select on plain `bool` conditions.
        assert_eq!(f64::select(&true, &2.0, &3.0), Ok(2.0));
        assert_eq!(f32::select(&false, &2.0, &3.0), Ok(3.0));

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 3 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[branch_type.clone(), branch_type.clone(), branch_type.clone()]),
            Err(TypeError { message: "select condition data type f64 is not bool".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])),
                branch_type.clone(),
                branch_type.clone(),
            ]),
            Err(TypeError { message: "select condition shape [2] differs from on_true shape [3]".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                condition_type.clone(),
                branch_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
            ]),
            Err(TypeError { message: "select on_true shape [3] differs from on_false shape [2]".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                condition_type.clone(),
                branch_type.clone(),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
            ]),
            Err(TypeError { message: "select on_true data type f64 differs from on_false data type f32".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, SelectOperation>::new();
        let program_condition = builder.add_input(condition_type);
        let program_on_true = builder.add_input(branch_type.clone());
        let program_on_false = builder.add_input(branch_type);
        let program_output = builder
            .add_instruction(operation, vec![program_condition, program_on_true, program_on_false])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, TestArray>(
                vec![program_output],
                vec![Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[3], %1:f64[3], %2:f64[3] .
                let %3:f64[3] = select %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }
}
