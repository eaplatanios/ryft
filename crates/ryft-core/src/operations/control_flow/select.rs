use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, ArrayType, Broadcastable, DataType};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::ElementwiseOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProvider, ProgramError, RegionInterface, Type, TypeError, Typed,
    Value,
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

impl<T: Type> SelectOperation<T> {
    /// Constructs a select operation for the `T` type universe.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Copy for SelectOperation<T> {}

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
                "`{}` condition data type `{}` is not `{}`",
                SELECT_OPERATION_NAME,
                input_types[0],
                DataType::Boolean,
            )));
        }

        // The two branch data types are promoted together (the Boolean condition is a mask, not a value that promotes
        // into the result), so `select` supports mixed-but-promotable branch data types like JAX's `jnp.where`.
        input_types[1].broadcast(&input_types[2]).map(|output| vec![output]).map_err(|_| {
            TypeError::invalid(format!("`{SELECT_OPERATION_NAME}` input types are not broadcast-compatible"))
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
                "`{}` condition data type `{}` is not `{}`",
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
            let mut primal = context.primal().bind(
                SelectOperation::new(),
                Vec::new(),
                &[condition.primal().clone(), on_true.primal().clone(), on_false.primal().clone()],
            )?;
            check_count!("output", primal, 1, ProgramError);
            let primal = primal.remove(0);
            let tangent = if on_true.tangent().is_zero() && on_false.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent()?)
            } else {
                // A `select` needs both branch tangents as real values, so materialize the structurally zero side.
                let on_true_tangent = on_true.tangent().clone().materialize(context.tangent())?;
                let on_false_tangent = on_false.tangent().clone().materialize(context.tangent())?;
                let condition = context.primal_to_tangent(condition.primal().clone())?;
                let exemplar = context.primal_to_tangent(primal.clone())?;
                let mut tangents = context.tangent().bind(
                    SelectOperation::new(),
                    Vec::new(),
                    &[condition, on_true_tangent, on_false_tangent],
                )?;
                check_count!("output", tangents, 1, ProgramError);
                let output_tangent_type = primal.r#type().tangent()?;
                MaybeZero::Value(tangents.remove(0).align_tangent(&output_tangent_type, &exemplar)?)
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<ZeroLikeOperation<V::Type>> + OperationProvider<V::Type, ZeroOperation<V::Type>, Operation = O> + From<SelectOperation<V::Type>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
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
            check_count!("accumulator", accumulators, 3, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(cotangent) => {
                    // The condition is the known operand. The dispatch guarantees a `Known` operand
                    // carries its pullback value, so read the tracer directly.
                    let condition = inputs[0]
                        .as_known()
                        .expect("dispatch guarantees a known operand carries its pullback value")
                        .clone();
                    let cotangent_type = cotangent.r#type().into_owned();
                    let zero = if cotangent_type.identities().next().is_some() {
                        // A type with symbolic extents does not contain concrete runtime extents, so use the live
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
                        MaybeZero::Zero(cotangent_type).materialize(&**context)?
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
                    let on_true_type = inputs[1].r#type().cotangent()?;
                    let on_false_type = inputs[2].r#type().cotangent()?;
                    {
                        let contribution =
                            MaybeZero::Value(on_true.into_iter().next().unwrap().unalign_cotangent(&on_true_type)?);
                        accumulators[1].accumulate(context, contribution)?;
                        let contribution =
                            MaybeZero::Value(on_false.into_iter().next().unwrap().unalign_cotangent(&on_false_type)?);
                        accumulators[2].accumulate(context, contribution)?;
                        Ok(())
                    }
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
/// # use ryft_core::arrays::Array;
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

impl Select for Array {
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        // Mirrors the broadcasting `SelectOperation` type-inference contract: the condition must be Boolean-typed,
        // the three operand shapes broadcast together, and the two branch data types promote together to the output
        // data type. The condition is retyped to a branch data type before broadcasting so its Boolean data type
        // acts as a mask rather than promoting into the output.
        assert_eq!(condition.r#type().data_type(), DataType::Boolean, "select condition must have a Boolean data type");
        let output_type = ArrayType::broadcasted(&[
            condition.r#type().into_owned().with_data_type(on_true.r#type().data_type()),
            on_true.r#type().into_owned(),
            on_false.r#type().into_owned(),
        ])
        .map_err(|error| TypeError::invalid(error.to_string()))?;

        // Convert only when promotion requires it. Equal-typed branches retain their original physical storage and
        // arbitrary layouts; conversion remains responsible for the element semantics until its own typed-byte slice.
        let output_data_type = output_type.data_type();
        let on_true = on_true.promoted_to(output_data_type)?;
        let on_false = on_false.promoted_to(output_data_type)?;

        let output_shape = output_type.static_shape().unwrap();
        let condition_shape = condition.r#type().static_shape().unwrap();
        let true_shape = on_true.r#type().static_shape().unwrap();
        let false_shape = on_false.r#type().static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let condition_strides = condition_shape.row_major_strides();
        let true_strides = true_shape.row_major_strides();
        let false_strides = false_shape.row_major_strides();
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let condition_addressing = ArrayAddressing::new(condition.r#type().into_owned())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type().into_owned())?;
        let false_addressing = ArrayAddressing::new(on_false.r#type().into_owned())?;
        let mut output_bytes = vec![0; output_addressing.storage_byte_len()];
        for output_index in 0..output_addressing.element_count() {
            let condition_index = Self::broadcast_index(
                output_index,
                &output_shape,
                &output_strides,
                &condition_shape,
                &condition_strides,
            );
            let condition_range = condition_addressing.byte_range_for_flat_index(condition_index);
            let (source, source_range) = if condition.storage_bytes()[condition_range.start] != 0 {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &true_shape, &true_strides);
                (on_true.storage_bytes(), true_addressing.byte_range_for_flat_index(source_index))
            } else {
                let source_index =
                    Self::broadcast_index(output_index, &output_shape, &output_strides, &false_shape, &false_strides);
                (on_false.storage_bytes(), false_addressing.byte_range_for_flat_index(source_index))
            };
            let output_range = output_addressing.byte_range_for_flat_index(output_index);
            output_bytes[output_range].copy_from_slice(&source[source_range]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(output_bytes)))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Dimension, Layout, Shape, StridedLayout};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::programs::{EmptyRegionDriver, ProgramError, Typed};

    use super::*;

    #[test]
    fn test_select() {
        let array_operation = SelectOperation::<ArrayType>::new();

        // Check operation identity in the array type universe.
        assert_eq!(array_operation.name(), SELECT_OPERATION_NAME);
        assert_eq!(format!("{array_operation}"), SELECT_OPERATION_NAME);
    }

    #[test]
    fn test_select_type_inference() {
        let array_operation = SelectOperation::<ArrayType>::new();

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
                    error = "`select` condition data type `f64` is not `bool`",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])),
                        branch_type.clone(),
                        branch_type.clone(),
                    ],
                    error = "`select` input types are not broadcast-compatible",
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
    }

    #[test]
    fn test_select_interpretation() {
        let branch_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        // Check eager selection, scalar broadcasting, and mixed branch data-type promotion together.
        let output = SelectOperation::<ArrayType>::new()
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::vector(vec![true, false, true]),
                    Array::scalar(7.0_f32),
                    Array::vector(vec![4.0_f64, 5.0, 6.0]),
                ],
            )
            .unwrap()
            .remove(0);
        assert_eq!(output.r#type().into_owned(), branch_type);
        assert_eq!(output.to_f64s(), vec![7.0, 5.0, 7.0]);
    }

    #[test]
    fn test_select_partial_evaluation() {
        // Check that known inputs fold and unknown inputs residualize.
        check_operation_partial_evaluation!(
            operation = SelectOperation::<ArrayType>::new(),
            inputs = [Array::scalar(true), Array::scalar(2.0_f32), Array::scalar(3.0_f64)],
            expected = Array::scalar(2.0_f64),
        );
    }

    #[test]
    fn test_select_batching() {
        let array_operation = SelectOperation::<ArrayType>::new();

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
    }

    #[test]
    fn test_select_differentiation() {
        // Check that differentiation routes tangents and cotangents through the selected branch. This stays explicit
        // because the operation-check helper's finite-difference oracle cannot perturb a Boolean condition input.
        fn piecewise<V: Clone + Compare<V> + Select + std::ops::Add<Output = V>>(
            x: V,
            y: V,
        ) -> Result<V, ProgramError> {
            let mask = x.compare(&y, ComparisonDirection::GreaterThan)?;
            Select::select(&mask, &(x.clone() + x.clone()), &(y.clone() + y.clone() + y.clone()))
        }

        let (primal, tangent) = differentiate_at((Array::scalar(3.0), Array::scalar(2.0)))
            .jvp((Array::scalar(1.0), Array::scalar(0.0)), |(x, y)| piecewise(x, y))
            .unwrap();
        assert_eq!(primal, Array::scalar(6.0));
        assert_eq!(tangent, Array::scalar(2.0));
        let (value, gradient) = differentiate_at((Array::scalar(3.0), Array::scalar(2.0)))
            .value_and_gradient(|(x, y)| piecewise(x, y).unwrap())
            .unwrap();
        assert_eq!(value, Array::scalar(6.0));
        assert_eq!(gradient.0, Array::scalar(2.0));
        assert_eq!(gradient.1, Array::scalar(0.0));

        let (primal, tangent) = differentiate_at((Array::scalar(1.0), Array::scalar(2.0)))
            .jvp((Array::scalar(0.0), Array::scalar(1.0)), |(x, y)| piecewise(x, y))
            .unwrap();
        assert_eq!(primal, Array::scalar(6.0));
        assert_eq!(tangent, Array::scalar(3.0));
        let (value, gradient) = differentiate_at((Array::scalar(1.0), Array::scalar(2.0)))
            .value_and_gradient(|(x, y)| piecewise(x, y).unwrap())
            .unwrap();
        assert_eq!(value, Array::scalar(6.0));
        assert_eq!(gradient.0, Array::scalar(0.0));
        assert_eq!(gradient.1, Array::scalar(3.0));
    }

    #[test]
    fn test_select_transposition() {
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

    #[test]
    fn test_array_select() {
        let condition = Array::vector(vec![true, false, true]);
        let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
        let on_false = Array::vector(vec![-1.0, -2.0, -3.0]);
        assert_eq!(Array::select(&condition, &on_true, &on_false).unwrap(), Array::vector(vec![1.0, -2.0, 3.0]));
        // The condition broadcasts against the branches, and the branch data types promote together.
        let broadcast =
            Array::select(&Array::scalar(true), &Array::vector(vec![1.0f32, 2.0]), &Array::vector(vec![-1.0f64, -2.0]))
                .unwrap();
        assert_eq!(broadcast, Array::vector(vec![1.0f64, 2.0]));

        // General broadcasting reads every input through its physical layout and writes one dense output without
        // converting equal-typed branch elements through an intermediate representation.
        let condition_type = ArrayType::new_static(DataType::Boolean, [2, 1])
            .with_layout(Layout::Strided(StridedLayout::new(vec![-3, 1])));
        let condition = Array::from_elements(condition_type, &[true, false]).unwrap();
        let true_type =
            ArrayType::new_static(DataType::U16, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, -2])));
        let on_true = Array::from_elements(true_type, &[0x1111u16, 0x2222, 0x3333]).unwrap();
        let false_type =
            ArrayType::new_static(DataType::U16, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-4, 2])));
        let on_false = Array::from_elements(false_type, &[0xaaaau16, 0xbbbb]).unwrap();
        let selected = Array::select(&condition, &on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &ArrayType::new_static(DataType::U16, [2, 3]));
        assert_eq!(selected.elements::<u16>(), Ok(vec![0x1111, 0x2222, 0x3333, 0xbbbb, 0xbbbb, 0xbbbb]),);
        assert_eq!(selected.storage_bytes(), [0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb],);
    }
}
