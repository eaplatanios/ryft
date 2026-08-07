use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{ArrayType, DataType};
use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::ElementwiseOperation;
use crate::operations::constants::zero::{Zero, ZeroOperationProvider};
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`SelectOperation`].
pub const SELECT_OPERATION_NAME: &str = "select";

/// [`Operation`] that performs an elementwise selection between two values driven by a Boolean condition.
///
/// The `T` parameter fixes the operation's type universe at construction time. Consequently,
/// `SelectOperation<DataType>` and `SelectOperation<ArrayType>` are distinct zero-sized payload types, and each payload
/// implements exactly one [`Operation`] contract. Refer to the documentation of [`Select`] for more information.
#[derive(Clone, Debug)]
pub struct SelectOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> Copy for SelectOperation<T> {}

impl<T: Type> SelectOperation<T> {
    /// Constructs a select operation for the `T` type universe.
    #[inline]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Display for SelectOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SELECT_OPERATION_NAME)
    }
}

impl Operation for SelectOperation<DataType> {
    type Type = DataType;

    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        if !input_types[0].is_boolean() {
            return Err(TypeError::invalid(format!(
                "'select' condition data type {} is not {}",
                input_types[0],
                DataType::Boolean
            )));
        }

        // The two branch data types are promoted together (the Boolean condition is a mask, not a value that promotes
        // into the result), so `select` supports mixed-but-promotable branch data types like JAX's `jnp.where`.
        input_types[1].broadcast(&input_types[2]).map(|output| vec![output]).map_err(|_| {
            TypeError::invalid(format!("'{SELECT_OPERATION_NAME}' input types are not broadcast-compatible"))
        })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME).map(|_| ())
    }
}

impl Operation for SelectOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        ElementwiseOperation::infer_output_types(self, input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME).map(|_| ())
    }
}

// [`SelectOperation`] is a broadcasting elementwise operation. `select(condition, on_true, on_false)` selects per
// element between the two branches under the Boolean `condition`, broadcasting the three operands' shapes together like
// [JAX's `jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html). Its type inference therefore
// overrides the plain elementwise default. The condition must be `DataType::Boolean` and the two branches' `DataType`s
// are promoted together (i.e., the condition is a mask, not a value that promotes into the result, so that
// `select(condition, f32, f64)` yields an `f64` result like JAX's `jnp.where`), while the output `Shape` is the
// broadcast of all three operand shapes and the output `DataType` is the promotion of the two branch data types.
// Implementing `ElementwiseOperation` also gives `select` the standard elementwise batching rule through its blanket
// `BatchableOperation` implementation.
impl ElementwiseOperation for SelectOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        3
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        let (condition, on_true, on_false) = (&input_types[0], &input_types[1], &input_types[2]);
        if !condition.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "'{}' condition data type {} is not {}",
                SELECT_OPERATION_NAME,
                condition.data_type(),
                DataType::Boolean,
            )));
        }

        // Broadcast the three operand shapes together and promote the two branch data types, retyping the Boolean
        // condition to a branch data type first so it acts as a mask rather than a value that promotes into the result.
        // The output shape and placement are then the standard elementwise broadcast of all three operands, and the
        // output data type is the promotion of the two branch data types.
        let condition = condition.clone().with_data_type(on_true.data_type());
        Ok(vec![self.infer_elementwise_broadcast_type(&[condition, on_true.clone(), on_false.clone()])?])
    }
}

impl<T: Type, C: Domain<Type = T, Value: Select>> InterpretableOperation<C> for SelectOperation<T>
where
    Self: Operation<Type = C::Type>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Interpretation selects through the value-level `Select` capability. The condition is an ordinary value in
        // the active domain: concrete arrays use themselves as the Boolean mask, and context-carrying values such as
        // staged `Tracer`s bind a `SelectOperation` through their own context.
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![C::Value::select(&inputs[0], &inputs[1], &inputs[2])?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<SelectOperation<T>>>> PartiallyEvaluatableOperation<C>
    for SelectOperation<T>
where
    Self: Operation<Type = T>,
{
}

macro_rules! impl_select_differentiation {
    // This branch attaches one shared rule declaration to both concrete select operation contracts.
    ($($rules:tt)*) => {
        impl_differentiable_operation! {
            SelectOperation<DataType>,
            $($rules)*
        }
        impl_differentiable_operation! {
            SelectOperation<ArrayType>,
            $($rules)*
        }
    };
}

impl_select_differentiation! {
    jvp<C>
    where
        C: Zero<C::Value>,
        C::Type: DifferentiableType,
        C::Value: ElementwiseDerivativeAlignment<C::Type>,
        C::Operation: From<SelectOperation<C::Type>>,
    {
        |_operation, context, _driver, inputs| {
            // Forward-mode differentiation rule for `SelectOperation`. The primal output is `select(condition, on_true,
            // on_false)` over the input primals, and the tangent selects the branch tangents under the *same* primal
            // condition (i.e., a `select` is piecewise linear in its branches), with the condition carried as an
            // ordinary primal operand edge. When both branch tangents are structural zeros, the output tangent is a
            // structural zero of the output type.
            check_count!("input", inputs, 3, ProgramError);
            let condition = &inputs[0];
            let on_true = &inputs[1];
            let on_false = &inputs[2];
            let mut primal = context.bind(
                SelectOperation::new(),
                Vec::new(),
                &[condition.primal().clone(), on_true.primal().clone(), on_false.primal().clone()],
            )?;
            check_count!("output", primal, 1, ProgramError);
            let primal = primal.remove(0);
            let tangent = if on_true.tangent().is_zero() && on_false.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent())
            } else {
                // A `select` needs both branch tangents as real values, so materialize the structurally zero side.
                let on_true_tangent = on_true.tangent().clone().materialize(context)?;
                let on_false_tangent = on_false.tangent().clone().materialize(context)?;
                let mut tangents = context.bind(
                    SelectOperation::new(),
                    Vec::new(),
                    &[condition.primal().clone(), on_true_tangent, on_false_tangent],
                )?;
                check_count!("output", tangents, 1, ProgramError);
                let output_tangent_type = primal.r#type().tangent();
                MaybeZero::Value(tangents.remove(0).align_tangent(&output_tangent_type)?)
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<ZeroLikeOperation<V::Type>> + ZeroOperationProvider<V::Type> + From<SelectOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, context, _driver, inputs, outputs| {
            // Partition-aware transposition rule for `SelectOperation`. The Boolean condition (i.e., operand 0) has no
            // tangent space, and so in a valid pushforward it is the known operand and the two branches (i.e., operands
            // 1 and 2) are the linear ones. The forward map `(on_true, on_false) ↦ select(condition, on_true,
            // on_false)` routes the output cotangent into the branch the known condition selected: the `on_true`
            // cotangent is `select(condition, cotangent, 0)` and the `on_false` cotangent is `select(condition, 0,
            // cotangent)`, each staged as a primal `select` over the condition read from the pullback through the known
            // operand's value. The condition receives a structural zero, and a zero output cotangent stays a structural
            // zero. The rule is generic over the primary type `V::Type` because it only reaches the branch type (i.e.,
            // `input_types[1]`), the known condition operand value, and the primal `select`; it carries no rank- or
            // shape-specific logic, so it applies uniformly to every operation family that contains `SelectOperation`.
            check_count!("input", inputs, 3, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(inputs
                    .iter()
                    .map(|input| MaybeZero::Zero(input.r#type().cotangent()))
                    .collect()),
                MaybeZero::Value(cotangent) => {
                    // The condition is the known operand. The dispatch guarantees a `Known` operand
                    // carries its pullback value, so read the tracer directly.
                    let condition = inputs[0]
                        .as_known()
                        .expect("dispatch guarantees a known operand carries its pullback value")
                        .clone();
                    let cotangent_type = cotangent.r#type().into_owned();
                    let zero = if cotangent_type.identities().next().is_some() {
                        // An identity-bearing type does not contain concrete runtime extents, so use the live
                        // cotangent as the shape exemplar. Identity-free types retain the canonical nullary zero,
                        // whose zero-producing marker keeps higher-order partial evaluation structural.
                        let mut zero = context.stage_operation(
                            ZeroLikeOperation::new(),
                            Vec::new(),
                            std::slice::from_ref(cotangent),
                        )?;
                        check_count!("output", zero, 1, ProgramError);
                        zero.remove(0)
                    } else {
                        MaybeZero::Zero(cotangent_type).materialize(context)?
                    };
                    let on_true = context.stage_operation(
                        SelectOperation::new(),
                        Vec::new(),
                        &[condition.clone(), cotangent.clone(), zero.clone()],
                    )?;
                    check_count!("output", on_true, 1, ProgramError);
                    let on_false = context.stage_operation(
                        SelectOperation::new(),
                        Vec::new(),
                        &[condition, zero, cotangent.clone()],
                    )?;
                    check_count!("output", on_false, 1, ProgramError);
                    let on_true_type = inputs[1].r#type().cotangent();
                    let on_false_type = inputs[2].r#type().cotangent();
                    Ok(vec![
                        MaybeZero::Zero(inputs[0].r#type().cotangent()),
                        MaybeZero::Value(on_true.into_iter().next().unwrap().unalign_cotangent(&on_true_type)?),
                        MaybeZero::Value(on_false.into_iter().next().unwrap().unalign_cotangent(&on_false_type)?),
                    ])
                }
            }
        }
    },
}

/// Represents the ability to perform an elementwise selection between two values driven by a condition. This is the
/// direct analogue of JAX's [`jnp.where`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) in its
/// three-argument form.
///
/// For arrays, `Self::select(condition, on_true, on_false)` returns a value whose `i`-th element equals `on_true`'s
/// `i`-th element when the corresponding element of `condition` is true, and `on_false`'s otherwise. The three operand
/// shapes broadcast together and the two branch data types promote together, so `condition`, `on_true`, and `on_false`
/// need not share a shape and the branches need not share a data type. The condition is represented by the same value
/// type as the branches: concrete arrays use Boolean-typed condition arrays, and staged [`Tracer`]s use Boolean-typed
/// tracer values.
///
/// # Example
///
/// The following example shows how to use [`Select`] in practice:
///
/// ```rust
/// # use ryft_core::operations::control_flow::Select;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Array values pair with a Boolean-typed condition array of the same shape.
/// let condition = Array::vector(vec![true, false, true]);
/// let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
/// let on_false = Array::vector(vec![4.0, 5.0, 6.0]);
/// let output = Array::select(&condition, &on_true, &on_false)?;
/// assert_eq!(output.to_f64s(), vec![1.0, 5.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub trait Select: Sized {
    /// Selects from `on_true` and `on_false` based on `condition`. Refer to the documentation of this
    /// trait for more information on what this operation does.
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError>;
}

// Any context-carrying value selects by binding a [`SelectOperation`] through its own context. A staged tracer records
// the operation, a batching tracer selects the packed values under the common batch axis, and a differentiation dual
// selects the primals and (linearly) the tangents by the same condition. The `From<SelectOperation<V::Type>>` bound
// makes this blanket disjoint from the concrete eager value types (whose context operation is `ConstantOperation`),
// which implement `Select` directly.
impl<V: Value> Select for V
where
    V::DispatchDomain: Context<Operation: From<SelectOperation<V::Type>>>,
{
    #[inline]
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        Ok(condition
            .dispatch_domain()
            .bind(SelectOperation::new(), Vec::new(), &[condition.clone(), on_true.clone(), on_false.clone()])?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Dimension, Shape};
    use crate::backends::Array;
    use crate::differentiation::{jvp, value_and_gradient};
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::programs::{ProgramError, Typed};

    use super::*;

    #[test]
    fn test_select() {
        let array_operation = SelectOperation::<ArrayType>::new();

        // Check operation identity in the array type universe.
        assert_eq!(array_operation.name(), SELECT_OPERATION_NAME);
        assert_eq!(format!("{array_operation}"), SELECT_OPERATION_NAME);

        // Check ternary shape broadcasting and branch promotion in the array type universe.
        let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let branch_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let scalar_branch = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]));
        let scalar_condition = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(1)]));
        check_operation_type_inference!(
            operation = array_operation,
            cases = [
                {
                    input_types = [condition_type.clone(), branch_type.clone(), branch_type.clone()],
                    output_types = [branch_type.clone()],
                },
                {
                    input_types = [condition_type.clone(), scalar_branch.clone(), branch_type.clone()],
                    output_types = [branch_type.clone()],
                },
                {
                    input_types = [scalar_condition, branch_type.clone(), branch_type.clone()],
                    output_types = [branch_type.clone()],
                },
                {
                    type = ArrayType,
                    input_types = [],
                    error = "expected 3 inputs but got 0",
                },
                {
                    input_types = [branch_type.clone(), branch_type.clone(), branch_type.clone()],
                    error = "'select' condition data type f64 is not bool",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])),
                        branch_type.clone(),
                        branch_type.clone(),
                    ],
                    error = "'select' input types are not broadcast-compatible",
                },
                {
                    input_types = [
                        condition_type.clone(),
                        branch_type.clone(),
                        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])),
                    ],
                    output_types = [branch_type.clone()],
                },
            ],
        );

        // Check eager selection, scalar broadcasting, and mixed branch data-type promotion together.
        let output = Select::select(
            &Array::vector(vec![true, false, true]),
            &Array::scalar(7.0_f32),
            &Array::vector(vec![4.0_f64, 5.0, 6.0]),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), branch_type);
        assert_eq!(output.to_f64s(), vec![7.0, 5.0, 7.0]);

        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = SelectOperation::<ArrayType>::new(),
            inputs = [Array::scalar(true), Array::scalar(2.0_f32), Array::scalar(3.0_f64)],
            expected = Array::scalar(2.0_f64),
        );

        // Check elementwise batching with mapped conditions and a replicated branch.
        check_operation_batching!(
            @exact,
            operation = array_operation,
            axis_size = 2,
            cases = [{
                inputs = [
                    (@mapped(axis = 0), Array::vector(vec![true, false])),
                    (@replicated, Array::scalar(2.0)),
                    (@mapped(axis = 0), Array::vector(vec![3.0, 4.0])),
                ],
                outputs = [(@mapped(axis = 0), Array::vector(vec![2.0, 4.0]))],
            }],
        );

        // Check that differentiation routes tangents and cotangents through the selected branch. This stays explicit
        // because the operation-check helper's finite-difference oracle cannot perturb a Boolean condition input.
        fn piecewise<V: Clone + Compare<V> + Select + std::ops::Add<Output = V>>(
            x: V,
            y: V,
        ) -> Result<V, ProgramError> {
            let mask = x.compare(&y, ComparisonDirection::GreaterThan)?;
            Select::select(&mask, &(x.clone() + x.clone()), &(y.clone() + y.clone() + y.clone()))
        }

        let (primal, tangent) = jvp(
            |(x, y)| piecewise(x, y),
            (Array::scalar(3.0), Array::scalar(2.0)),
            (Array::scalar(1.0), Array::scalar(0.0)),
        )
        .unwrap();
        assert_eq!(primal, Array::scalar(6.0));
        assert_eq!(tangent, Array::scalar(2.0));
        let (value, gradient) =
            value_and_gradient(|(x, y)| piecewise(x, y).unwrap(), (Array::scalar(3.0), Array::scalar(2.0))).unwrap();
        assert_eq!(value, Array::scalar(6.0));
        assert_eq!(gradient.0, Array::scalar(2.0));
        assert_eq!(gradient.1, Array::scalar(0.0));

        let (primal, tangent) = jvp(
            |(x, y)| piecewise(x, y),
            (Array::scalar(1.0), Array::scalar(2.0)),
            (Array::scalar(0.0), Array::scalar(1.0)),
        )
        .unwrap();
        assert_eq!(primal, Array::scalar(6.0));
        assert_eq!(tangent, Array::scalar(3.0));
        let (value, gradient) =
            value_and_gradient(|(x, y)| piecewise(x, y).unwrap(), (Array::scalar(1.0), Array::scalar(2.0))).unwrap();
        assert_eq!(value, Array::scalar(6.0));
        assert_eq!(gradient.0, Array::scalar(0.0));
        assert_eq!(gradient.1, Array::scalar(3.0));

        // Check that primitive transposition partitions the cotangent between the two linear branches.
        let condition = Array::vector(vec![true, false]);
        let on_true = Array::vector(vec![10.0, 20.0]);
        let cotangent = Array::vector(vec![5.0, 7.0]);
        let branch_type = on_true.r#type().into_owned();
        check_operation_transposition!(
            @exact,
            operation = SelectOperation::<ArrayType>::new(),
            cases = [{
                inputs = [
                    (@known, condition),
                    (@linear(type = branch_type.clone())),
                    (@linear(type = branch_type)),
                ],
                output_cotangents = [cotangent],
                input_cotangents = [Array::vector(vec![5.0, 0.0]), Array::vector(vec![0.0, 7.0])],
            }],
        );
    }
}
