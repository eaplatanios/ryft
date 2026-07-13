use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::Context;
use crate::contexts::Domain;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::types::{ArrayType, DataType, TypeError};

// TODO(eaplatanios): Review this module.

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
                message: format!("'select' condition data type {} is not {}", input_types[0], DataType::Boolean),
            });
        }
        // The two branch data types are promoted together (the Boolean condition is a mask, not a value that promotes
        // into the result), so `select` supports mixed-but-promotable branch data types like `jnp.where`.
        input_types[1]
            .broadcast(&input_types[2])
            .map(|output| vec![output])
            .map_err(|_| TypeError { message: "'select' input types are not broadcast-compatible".to_string() })
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

    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME).map(|_| ())
    }
}

/// [`SelectOperation`] is a broadcasting elementwise operation: `select(condition, on_true, on_false)` selects
/// per element between the two branches under the Boolean `condition`, broadcasting the three operands' shapes
/// together like [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html). Its type inference
/// therefore overrides the plain elementwise default: the condition must be [`DataType::Boolean`] and the two
/// branches' [`DataType`]s are promoted together (the condition is a mask, not a value that promotes into the
/// result, so `select(condition, f32, f64)` yields an `f64` result like `jnp.where`), while the output
/// [`Shape`](crate::types::Shape) is the broadcast of all three operand shapes and the output [`DataType`] is the
/// promotion of the two branch data types. Implementing [`ElementwiseOperation`] also gives `select` the standard
/// elementwise batching rule through the blanket [`BatchableOperation`](crate::BatchableOperation) implementation.
impl ElementwiseOperation for SelectOperation {
    #[inline]
    fn input_count(&self) -> usize {
        3
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        let (condition, on_true, on_false) = (&input_types[0], &input_types[1], &input_types[2]);
        if condition.data_type() != DataType::Boolean {
            return Err(TypeError {
                message: format!("'select' condition data type {} is not {}", condition.data_type(), DataType::Boolean),
            });
        }
        // Broadcast the three operand shapes together and promote the two branch data types, retyping the Boolean
        // condition to a branch data type first so it acts as a mask rather than a value that promotes into the
        // result. The output shape and placement are then the standard elementwise broadcast of all three operands
        // and the output data type is the promotion of the two branch data types.
        let condition = condition.clone().with_data_type(on_true.data_type());
        Ok(vec![self.broadcast_output_type(&[condition, on_true.clone(), on_false.clone()])?])
    }
}

/// Interpretation selects through the value-level [`Select`] capability, with the condition operand's
/// [`SelectCondition`] view providing the condition representation of the active value semantics: eager scalar values
/// decode the in-band Boolean into a plain [`bool`], eager array values pass themselves as the Boolean mask, and
/// context-carrying values (e.g., staged [`Tracer`](crate::Tracer)s) select by binding a [`SelectOperation`] through
/// their own context.
impl<V, C> InterpretableOperation<V, C> for SelectOperation
where
    V: Value + SelectCondition + Select<Condition = <V as SelectCondition>::Condition>,
    Self: Operation<V::Type>,
{
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![V::select(&inputs[0].select_condition()?, &inputs[1], &inputs[2])?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context> PartiallyEvaluatableOperation<C> for SelectOperation where C::Operation: From<SelectOperation> {}

/// Represents the ability to perform an elementwise selection between two values driven by a condition. This is the
/// direct analogue of JAX's [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) in its
/// three-argument form.
///
/// For arrays, `Self::select(condition, on_true, on_false)` returns a value whose `i`-th element equals `on_true`'s
/// `i`-th element when the corresponding element of `condition` is true, and `on_false`'s otherwise. The three
/// operand shapes broadcast together and the two branch data types promote together, so `condition`, `on_true`, and
/// `on_false` need not share a shape and the branches need not share a data type (see [`SelectOperation`]). The
/// condition and branch value types may differ: for scalar values the condition is a plain [`bool`], while array
/// value types pair with a Boolean-typed condition array. Value types that participate in closed staged operation
/// sets (e.g., [`Tracer`]) use `Condition = Self`, representing the condition as a [`DataType::Boolean`] value.
///
/// # Example
///
/// The following example shows how to use [`Select`] in practice:
///
/// ```rust
/// # use ryft_core::operations::control_flow::Select;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::scalars::Scalar;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Scalar values use a plain `bool` condition.
/// assert_eq!(Scalar::select(&true, &Scalar::from(2.0), &Scalar::from(3.0))?, Scalar::from(2.0));
/// assert_eq!(Scalar::select(&false, &Scalar::from(2.0), &Scalar::from(3.0))?, Scalar::from(3.0));
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

// TODO(eaplatanios): Figure out whether we actually need this.
/// Extracts the [`Select`] condition carried by a value: for value-condition domains (arrays, staged tracers) the
/// condition is the value itself, while for scalar domains it is the decoded in-band Boolean.
///
/// The condition of a [`SelectOperation`] crosses the primal/tangent boundary differently per domain: array and
/// staged [`Tracer`] values implement [`Select`] with `Condition = Self`, whereas eager scalar values implement it
/// with `Condition = bool` by decoding an in-band Boolean via [`BooleanLike::boolean`]. This trait gives the generic
/// JVP rule of [`SelectOperation`] a single hook to obtain the right [`Select`] condition from a primal value (and
/// from the captured-condition factor that the linear select interprets) without committing to one domain's condition
/// representation.
pub trait SelectCondition {
    /// The condition type accepted by this value's [`Select`] implementation.
    type Condition;

    /// Extracts this value's [`Select`] condition.
    fn select_condition(&self) -> Result<Self::Condition, ProgramError>;
}

/// Any context-carrying value selects by binding a [`SelectOperation`] through its own context: a staged tracer
/// records the operation, a batching tracer selects the packed values under the common batch axis, and a JVP dual
/// selects the primals and (linearly) the tangents by the same condition. The `From<SelectOperation>` bound makes
/// this blanket disjoint from the concrete eager value types (whose context operation is
/// [`ConstantOperation`](crate::operations::constants::ConstantOperation)), which implement [`Select`] directly.
impl<V: Value> Select for V
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<SelectOperation>,
{
    type Condition = Self;

    #[inline]
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let mut outputs = condition.dispatch_domain().bind(
            SelectOperation,
            &[],
            &[],
            &[condition.clone(), on_true.clone(), on_false.clone()],
        )?;
        Ok(outputs.remove(0))
    }
}

/// For context-carrying values, the [`Select`] condition is the value itself.
impl<V: Value> SelectCondition for V
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<SelectOperation>,
{
    type Condition = Self;

    #[inline]
    fn select_condition(&self) -> Result<Self, ProgramError> {
        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::operations::BooleanLike;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tests::TestArray;
    use crate::types::{Shape, Size, Typed};

    use super::*;

    #[test]
    fn test_select() {
        let operation = SelectOperation;

        // Operation identity.
        assert_eq!(Operation::<ArrayType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), SELECT_OPERATION_NAME);

        // Scalar (`DataType`) type inference validates the Boolean condition and promotes the two branch data types.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::Boolean, DataType::F64, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::F64, DataType::F64, DataType::F64]),
            Err(TypeError { message: "'select' condition data type f64 is not bool".to_string() }),
        );
        // Mixed-but-promotable branch data types promote to their common type (`jnp.where`-style).
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[DataType::Boolean, DataType::F32, DataType::F64]),
            Ok(vec![DataType::F64]),
        );
        // Non-promotable branch data types are rejected.
        assert_eq!(
            Operation::<DataType>::infer_output_types(
                &operation,
                &[DataType::Boolean, DataType::F8E3M4, DataType::F32]
            ),
            Err(TypeError { message: "'select' input types are not broadcast-compatible".to_string() }),
        );

        // Scalar interpretation treats the in-band condition as true exactly when it is nonzero.
        let branches = [Scalar::from(2.0), Scalar::from(3.0)];
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &[Scalar::from(1.0), branches[0], branches[1]]),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &[Scalar::from(0.0), branches[0], branches[1]]),
            Ok(vec![Scalar::from(3.0)]),
        );

        // Scalar values decode their in-band Boolean payload through `BooleanLike` and reinterpret it as a
        // genuinely Boolean-typed `Scalar::Bool`.
        assert_eq!(Scalar::from(1.5).boolean(), Ok(true));
        assert_eq!(Scalar::from(0.0).boolean(), Ok(false));
        assert_eq!(Scalar::from(2.0).as_boolean(), Scalar::from(true));
        assert_eq!(Scalar::from(0.0f32).as_boolean(), Scalar::from(false));
        assert_eq!(Scalar::from(f16::from_f64(-3.0)).as_boolean(), Scalar::from(true));
        assert_eq!(Scalar::from(bf16::ZERO).boolean(), Ok(false));

        // Type inference validates the condition and branch types and returns the branch type.
        let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
        let branch_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[condition_type.clone(), branch_type.clone(), branch_type.clone()],
            ),
            Ok(vec![branch_type.clone()]),
        );

        // Type inference broadcasts the three operand shapes together (like the other elementwise operations),
        // keeping the branch data type: a size-1 branch broadcasts up to the condition/other-branch shape, and a
        // size-1 condition broadcasts up to the branch shape. The Boolean condition never promotes into the output
        // data type.
        let scalar_branch = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1)]));
        let scalar_condition = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(1)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[condition_type.clone(), scalar_branch.clone(), branch_type.clone()],
            ),
            Ok(vec![branch_type.clone()]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[scalar_condition, branch_type.clone(), branch_type.clone()],
            ),
            Ok(vec![branch_type.clone()]),
        );

        // Interpretation picks per-element between the two branches.
        let condition = TestArray::new(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let on_false = TestArray::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].values, vec![1.0, 5.0, 3.0]);

        // Interpretation broadcasts a size-1 branch up to the condition/other-branch shape, matching the broadcasting
        // type-inference contract.
        let condition = TestArray::new(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true = TestArray::new(scalar_branch.clone(), vec![7.0]);
        let on_false = TestArray::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].values, vec![7.0, 5.0, 7.0]);

        // Interpretation promotes mixed-but-promotable branch data types, so the output carries the promoted (`f64`)
        // data type of the two branches.
        let condition = TestArray::new(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true =
            TestArray::new(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])), vec![1.0, 2.0, 3.0]);
        let on_false = TestArray::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].values, vec![1.0, 5.0, 3.0]);

        // The scalar implementation selects on plain `bool` conditions.
        assert_eq!(Scalar::select(&true, &Scalar::from(2.0), &Scalar::from(3.0)), Ok(Scalar::from(2.0)));
        assert_eq!(Scalar::select(&false, &Scalar::from(2.0f32), &Scalar::from(3.0f32)), Ok(Scalar::from(3.0f32)));

        // Mixed-but-promotable branch data types promote the selected branch to the common type (`jnp.where`-style),
        // so the result carries the promoted data type regardless of which branch is selected.
        assert_eq!(Scalar::select(&true, &Scalar::from(2.0f32), &Scalar::from(3.0f64)), Ok(Scalar::from(2.0f64)));
        assert_eq!(Scalar::select(&false, &Scalar::from(2.0f32), &Scalar::from(3.0f64)), Ok(Scalar::from(3.0f64)));
        assert_eq!(Scalar::select(&true, &Scalar::from(2i32), &Scalar::from(3.0f64)), Ok(Scalar::from(2.0f64)));

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 3 inputs but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[branch_type.clone(), branch_type.clone(), branch_type.clone()],
            ),
            Err(TypeError { message: "'select' condition data type f64 is not bool".to_string() }),
        );
        // Genuinely incompatible shapes (neither dimension is 1) are rejected by the elementwise broadcast.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[
                    ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])),
                    branch_type.clone(),
                    branch_type.clone(),
                ]
            ),
            Err(TypeError { message: "'select' input types are not broadcast-compatible".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[
                    condition_type.clone(),
                    branch_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
                ]
            ),
            Err(TypeError { message: "'select' input types are not broadcast-compatible".to_string() }),
        );
        // Mixed-but-promotable branch data types promote to their common type at the broadcast shape.
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[
                    condition_type.clone(),
                    branch_type.clone(),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
                ]
            ),
            Ok(vec![branch_type.clone()]),
        );
        assert_eq!(
            InterpretableOperation::<TestArray, crate::EagerContext<TestArray>>::interpret(
                &operation,
                &crate::EagerContext::<TestArray>::new(),
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<TestArray, SelectOperation>::new();
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
