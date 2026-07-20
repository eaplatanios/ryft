use std::fmt::Display;

use crate::broadcasting::Broadcastable;
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;
use crate::differentiation::forward::DifferentiationDual;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`SelectOperation`].
pub const SELECT_OPERATION_NAME: &str = "select";

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

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
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
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
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

/// Interpretation selects through the value-level [`Select`] capability. The condition is an ordinary value in the
/// active domain: eager scalar values decode their in-band Boolean payload, eager array values use themselves as the
/// Boolean mask, and context-carrying values (e.g., staged [`Tracer`]s) bind a [`SelectOperation`] through their own
/// context.
impl<C: Domain> InterpretableOperation<C> for SelectOperation
where
    C::Value: Select,
    Self: Operation<C::Type>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![C::Value::select(&inputs[0], &inputs[1], &inputs[2])?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context> PartiallyEvaluatableOperation<C> for SelectOperation where C::Operation: From<SelectOperation> {}

impl_differentiable_elementwise_operation! {
    @custom
    SelectOperation,
    /// Forward-mode rule for [`SelectOperation`]: the primal output is `select(condition, on_true, on_false)` over
    /// the input primals, and the tangent selects the branch tangents under the *same* primal condition (a `select` is
    /// piecewise linear in its branches), with the condition carried as an ordinary primal operand edge. When both
    /// branch tangents are structural zeros, the output tangent is a structural zero of the output type.
    jvp<C>
    where
        C: Zero<C::Value>,
        C::Type: DifferentiableType,
        C::Operation: From<SelectOperation>,
        C::Value: ElementwiseDerivativeAlignment<C::Type>,
    {
        |_operation, context, _driver, inputs| {
            check_count!("input", inputs, 3, ProgramError);
            let condition = &inputs[0];
            let on_true = &inputs[1];
            let on_false = &inputs[2];
            // Bind the primal and tangent selects through the context rather than the value-level `Select` capability
            // because this rule already owns the active differentiation context and must preserve its tracing behavior.
            let mut primal = context.bind(
                SelectOperation,
                Vec::new(),
                &[condition.primal().clone(), on_true.primal().clone(), on_false.primal().clone()],
            )?;
            check_count!("output", primal, 1, ProgramError);
            let primal = primal.remove(0);
            let tangent = if on_true.tangent().is_zero() && on_false.tangent().is_zero() {
                MaybeZero::Zero(primal.r#type().tangent())
            } else {
                // A select needs both branch tangents as real values, so materialize the structurally zero side.
                let on_true_tangent = on_true.tangent().clone().materialize(context)?;
                let on_false_tangent = on_false.tangent().clone().materialize(context)?;
                let mut tangents = context.bind(
                    SelectOperation,
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

    /// Partition-aware transpose rule for [`SelectOperation`]. The Boolean condition (operand 0) has no tangent space,
    /// so in a valid pushforward it is the known operand and the two branches (operands 1 and 2) are the linear ones.
    /// The forward map `(on_true, on_false) ↦ select(condition, on_true, on_false)` routes the output cotangent into the
    /// branch the known condition selected: the `on_true` cotangent is `select(condition, cotangent, 0)` and the
    /// `on_false` cotangent is `select(condition, 0, cotangent)`, each staged as a primal `select` over the condition
    /// read from the pullback through the known operand's value. The condition receives a structural zero, and a zero
    /// output cotangent stays a structural zero.
    ///
    /// The rule is generic over the primary type `V::Type` because it only reaches the branch type (`input_types[1]`),
    /// the known condition operand value, and the primal `select`; it carries no rank- or shape-specific logic. It
    /// therefore applies to both the array [`ArrayOperation::Select`](crate::backends::arrays::ArrayOperation) and the
    /// scalar [`ScalarOperation::Select`](crate::backends::scalars::ScalarOperation) enum dispatch.
    transpose<V, O>
    where
        V::Type: DifferentiableType,
        O: From<ZeroOperation<V::Type>> + From<SelectOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
    {
        |_operation, context, _driver, inputs, outputs| {
            check_count!("input", inputs, 3, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(inputs
                    .iter()
                    .map(|input| {
                        let input_type = input.r#type();
                        MaybeZero::Zero(input_type.cotangent())
                    })
                    .collect()),
                MaybeZero::Value(cotangent) => {
                    // The condition is the known operand; the dispatch guarantees a `Known` operand carries its
                    // pullback value, so read the tracer directly.
                    let condition = inputs[0]
                        .as_known()
                        .expect("dispatch guarantees a known operand carries its pullback value")
                        .clone();
                    let zero = MaybeZero::Zero(cotangent.r#type().into_owned()).materialize(context)?;
                    let on_true = context.stage_operation(
                        SelectOperation,
                        Vec::new(),
                        &[condition.clone(), cotangent.clone(), zero.clone()],
                    )?;
                    check_count!("output", on_true, 1, ProgramError);
                    let on_false =
                        context.stage_operation(SelectOperation, Vec::new(), &[condition, zero, cotangent.clone()])?;
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
/// `i`-th element when the corresponding element of `condition` is true, and `on_false`'s otherwise. The three
/// operand shapes broadcast together and the two branch data types promote together, so `condition`, `on_true`, and
/// `on_false` need not share a shape and the branches need not share a data type (see [`SelectOperation`]). The
/// condition is represented by the same value type as the branches. Scalar values decode their in-band condition
/// through [`BooleanLike`](crate::operations::BooleanLike), while arrays use Boolean-typed condition arrays and staged
/// [`Tracer`]s use Boolean-typed tracer values.
///
/// # Example
///
/// The following example shows how to use [`Select`] in practice:
///
/// ```rust
/// # use ryft_core::operations::control_flow::Select;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::scalars::Scalar;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{ArrayType, DataType, Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Scalar values use an in-band Boolean scalar condition.
/// assert_eq!(Scalar::select(&Scalar::from(true), &Scalar::from(2.0), &Scalar::from(3.0))?, Scalar::from(2.0));
/// assert_eq!(Scalar::select(&Scalar::from(false), &Scalar::from(2.0), &Scalar::from(3.0))?, Scalar::from(3.0));
///
/// // Array values pair with a Boolean-typed condition array of the same shape.
/// let condition_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
/// let condition = Array::from_f64s(condition_type, vec![1.0, 0.0, 1.0]);
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
    #[inline]
    fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let mut outputs = condition.dispatch_domain().bind(
            SelectOperation,
            Vec::new(),
            &[condition.clone(), on_true.clone(), on_false.clone()],
        )?;
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::operations::BooleanLike;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::math::Add;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::ProgramError;
    use crate::programs::types::Typed;
    use crate::tracing_v2::{DenseDifferentiate, ForwardModeDifferentiate, ReverseModeDifferentiate, jacrev};
    use crate::types::{Shape, Size};

    use super::*;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over staged or differentiation-dual values of any context with
    /// [`Array`] semantics.
    fn piecewise_select<V>(x: V) -> V
    where
        V: Value<Type = ArrayType>,
        V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan).unwrap();
        let doubled = x.add(&x).unwrap();
        let tripled = doubled.add(&x).unwrap();
        Select::select(&mask, &doubled, &tripled).unwrap()
    }

    #[test]
    fn test_select() {
        let operation = SelectOperation;

        // Operation identity.
        assert_eq!(Operation::<ArrayType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::name(&operation), SELECT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), SELECT_OPERATION_NAME);

        // Scalar (`DataType`) type inference validates the Boolean condition and promotes compatible branch types.
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [DataType::Boolean, DataType::F64, DataType::F64],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::F64, DataType::F64, DataType::F64],
                    error = "'select' condition data type f64 is not bool",
                },
                {
                    input_types = [DataType::Boolean, DataType::F32, DataType::F64],
                    output_types = [DataType::F64],
                },
                {
                    input_types = [DataType::Boolean, DataType::F8E3M4, DataType::F32],
                    error = "'select' input types are not broadcast-compatible",
                },
            ],
        );

        // Scalar interpretation treats the in-band condition as true exactly when it is nonzero.
        let branches = [Scalar::from(2.0), Scalar::from(3.0)];
        assert_eq!(
            operation.interpret(
                &crate::EagerContext::<Scalar>::new(),
                &crate::EmptyRegionDriver,
                &[Scalar::from(1.0), branches[0], branches[1]]
            ),
            Ok(vec![Scalar::from(2.0)]),
        );
        assert_eq!(
            operation.interpret(
                &crate::EagerContext::<Scalar>::new(),
                &crate::EmptyRegionDriver,
                &[Scalar::from(0.0), branches[0], branches[1]]
            ),
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
        // Type inference broadcasts the three operand shapes together (like the other elementwise operations),
        // keeping the branch data type: a size-1 branch broadcasts up to the condition/other-branch shape, and a
        // size-1 condition broadcasts up to the branch shape. The Boolean condition never promotes into the output
        // data type.
        let scalar_branch = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1)]));
        let scalar_condition = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(1)]));
        check_operation_type_inference!(
            operation = operation,
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
                        ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])),
                        branch_type.clone(),
                        branch_type.clone(),
                    ],
                    error = "'select' input types are not broadcast-compatible",
                },
                {
                    input_types = [
                        condition_type.clone(),
                        branch_type.clone(),
                        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
                    ],
                    error = "'select' input types are not broadcast-compatible",
                },
                {
                    input_types = [
                        condition_type.clone(),
                        branch_type.clone(),
                        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])),
                    ],
                    output_types = [branch_type.clone()],
                },
            ],
        );

        // Interpretation picks per-element between the two branches.
        let condition = Array::from_f64s(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true = Array::vector(vec![1.0, 2.0, 3.0]);
        let on_false = Array::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<Array>::new(), &crate::EmptyRegionDriver, &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 5.0, 3.0]);

        // Interpretation broadcasts a size-1 branch up to the condition/other-branch shape, matching the broadcasting
        // type-inference contract.
        let condition = Array::from_f64s(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true = Array::from_f64s(scalar_branch.clone(), vec![7.0]);
        let on_false = Array::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<Array>::new(), &crate::EmptyRegionDriver, &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].to_f64s(), vec![7.0, 5.0, 7.0]);

        // Interpretation promotes mixed-but-promotable branch data types, so the output carries the promoted (`f64`)
        // data type of the two branches.
        let condition = Array::from_f64s(condition_type.clone(), vec![1.0, 0.0, 1.0]);
        let on_true =
            Array::from_f64s(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3)])), vec![1.0, 2.0, 3.0]);
        let on_false = Array::vector(vec![4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&crate::EagerContext::<Array>::new(), &crate::EmptyRegionDriver, &[condition, on_true, on_false])
            .unwrap();
        assert_eq!(*output[0].r#type(), branch_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 5.0, 3.0]);

        // The scalar implementation selects on in-band Boolean scalar conditions.
        assert_eq!(Scalar::select(&Scalar::from(true), &Scalar::from(2.0), &Scalar::from(3.0)), Ok(Scalar::from(2.0)),);
        assert_eq!(
            Scalar::select(&Scalar::from(false), &Scalar::from(2.0f32), &Scalar::from(3.0f32)),
            Ok(Scalar::from(3.0f32)),
        );

        // Mixed-but-promotable branch data types promote the selected branch to the common type (`jnp.where`-style),
        // so the result carries the promoted data type regardless of which branch is selected.
        assert_eq!(
            Scalar::select(&Scalar::from(true), &Scalar::from(2.0f32), &Scalar::from(3.0f64)),
            Ok(Scalar::from(2.0f64)),
        );
        assert_eq!(
            Scalar::select(&Scalar::from(false), &Scalar::from(2.0f32), &Scalar::from(3.0f64)),
            Ok(Scalar::from(3.0f64)),
        );
        assert_eq!(
            Scalar::select(&Scalar::from(true), &Scalar::from(2i32), &Scalar::from(3.0f64)),
            Ok(Scalar::from(2.0f64)),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            InterpretableOperation::<crate::EagerContext<Array>>::interpret(
                &operation,
                &crate::EagerContext::<Array>::new(),
                &crate::EmptyRegionDriver,
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );

        // Program rendering uses the canonical operation name.
        let mut builder = ProgramBuilder::<Array, SelectOperation>::new();
        let program_condition = builder.add_input(condition_type);
        let program_on_true = builder.add_input(branch_type.clone());
        let program_on_false = builder.add_input(branch_type);
        let program_output = builder
            .add_instruction(operation, Vec::new(), vec![program_condition, program_on_true, program_on_false])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder, Placeholder], Placeholder)
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

    #[test]
    fn test_select_jacfwd_computes_piecewise_derivative() {
        // Forward mode through `f(x) = select(x > 0, 2x, 3x)`: the tangent selects the branch tangents under the
        // same primal condition, so the derivative is 2 where x > 0 and 3 elsewhere.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|x| Ok(piecewise_select(x)), Array::scalar(2.0))
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(|x| Ok(piecewise_select(x)), Array::scalar(-2.0))
            .unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_scalar_select_jvp_and_gradient_flow_to_selected_branch() {
        // Differentiating `f(x, y) = select(x > y, 2x, 3y)` over `EagerContext<Scalar, ScalarOperation<Scalar>>`
        // exercises the scalar select rules: forward mode routes each branch tangent through the selected branch, and
        // reverse mode routes the cotangent there, so the derivative reaches only the selected branch's input.
        use crate::backends::scalars::ScalarOperation;

        fn piecewise<V>(x: V, y: V) -> Result<V, ProgramError>
        where
            V: Clone + Compare<Output = V> + Select + std::ops::Add<Output = V>,
        {
            let mask = x.compare(&y, ComparisonDirection::GreaterThan)?;
            Select::select(&mask, &(x.clone() + x.clone()), &(y.clone() + y.clone() + y.clone()))
        }

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        // `x > y`: the output is `2x`, so forward mode passes the `x` tangent through (scaled by 2) and zeroes the `y`
        // tangent, while the gradient is `(2, 0)`.
        let (primal, tangent) = domain
            .jvp(
                |(x, y)| piecewise(x, y),
                (Scalar::from(3.0), Scalar::from(2.0)),
                (Scalar::from(1.0), Scalar::from(0.0)),
            )
            .unwrap();
        assert_abs_diff_eq!(primal, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 2.0, epsilon = 1e-9);
        let (_, tangent) = domain
            .jvp(
                |(x, y)| piecewise(x, y),
                (Scalar::from(3.0), Scalar::from(2.0)),
                (Scalar::from(0.0), Scalar::from(1.0)),
            )
            .unwrap();
        assert_abs_diff_eq!(tangent, 0.0, epsilon = 1e-9);
        let (value, gradient) = domain
            .value_and_gradient(|(x, y)| piecewise(x, y).unwrap(), (Scalar::from(3.0), Scalar::from(2.0)))
            .unwrap();
        assert_abs_diff_eq!(value, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.0, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.1, 0.0, epsilon = 1e-9);

        // `x <= y`: the output is `3y`, so the roles flip; the gradient is `(0, 3)`.
        let (primal, tangent) = domain
            .jvp(
                |(x, y)| piecewise(x, y),
                (Scalar::from(1.0), Scalar::from(2.0)),
                (Scalar::from(1.0), Scalar::from(0.0)),
            )
            .unwrap();
        assert_abs_diff_eq!(primal, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 0.0, epsilon = 1e-9);
        let (_, tangent) = domain
            .jvp(
                |(x, y)| piecewise(x, y),
                (Scalar::from(1.0), Scalar::from(2.0)),
                (Scalar::from(0.0), Scalar::from(1.0)),
            )
            .unwrap();
        assert_abs_diff_eq!(tangent, 3.0, epsilon = 1e-9);
        let (value, gradient) = domain
            .value_and_gradient(|(x, y)| piecewise(x, y).unwrap(), (Scalar::from(1.0), Scalar::from(2.0)))
            .unwrap();
        assert_abs_diff_eq!(value, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.0, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.1, 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_computes_piecewise_derivative() {
        // Reverse mode through `f(x) = select(x > 0, 2x, 3x)` exercises the partition-aware select transpose: the
        // on_true cotangent is `select(condition, cotangent, 0)` and the on_false cotangent is
        // `select(condition, 0, cotangent)`.
        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_over_vector_masks_per_element() {
        // Per-element masking over a vector input: the Jacobian of `select(x > 0, 2x, 3x)` is diagonal with entries
        // 2 where x > 0 and 3 elsewhere.
        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::vector(vec![1.0, -1.0])).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[2]);
        assert_abs_diff_eq!(block.value().values()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[3], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_select_jacrev_unbroadcasts_mixed_precision_scalar_branches() {
        let scalar = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![5.0]);
        let f32_vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2)]));
        let vector =
            Array::from_f64s(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])), vec![2.0, -3.0]);

        let jacobian = jacrev(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &scalar, &vector)
            },
            (scalar.clone(), vector.clone()),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![1.0, 0.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![0.0, 0.0, 0.0, 1.0]);

        let jacobian = jacrev(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &vector, &scalar)
            },
            (scalar, vector),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![0.0, 1.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_select_partitioned_transpose_routes_the_cotangent_into_the_selected_branch() {
        // Condition `[true, false]` (known), branches and cotangent are length-two f64 vectors (linear branches).
        let condition = Array::vector(vec![1.0, 0.0]).as_boolean();
        // The branches are linear operands, so only their type enters the transpose; their values are unused.
        let on_true = Array::vector(vec![10.0, 20.0]);
        let cotangent = Array::vector(vec![5.0, 7.0]);
        let branch_type = on_true.r#type().into_owned();
        check_operation_transposition!(
            @exact,
            operation = SelectOperation,
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
