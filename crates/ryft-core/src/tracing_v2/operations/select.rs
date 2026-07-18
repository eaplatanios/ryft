use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::control_flow::{SELECT_OPERATION_NAME, Select, SelectCondition, SelectOperation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::interpretation::InterpretationDriver;
use crate::programs::types::{TypeError, Typed};
use crate::types::{ArrayType, DataType};

use crate::differentiation::elementwise::ElementwiseDerivativeAlignment;

/// Captured-condition select operation used in linear tangent and cotangent programs.
///
/// The ordinary [`SelectOperation`] is not the linear primitive because its operand list is
/// `(condition, on_true, on_false)`, and the Boolean condition is a primal value with no tangent space. The linear map
/// produced by the JVP of `select(condition, on_true, on_false)` instead fixes that primal condition as a captured
/// factor and acts only on the two branch tangents (or cotangents):
///
/// ```text
/// (on_true_tangent, on_false_tangent) -> select(condition, on_true_tangent, on_false_tangent)
/// ```
///
/// This operation stores that captured `condition` factor in the operation payload, which lets residualized
/// pushforwards remap or instantiate it like other captured factors. Its transpose routes the output cotangent into
/// the selected branch: `select(condition, cotangent, 0)` for the `on_true` input and
/// `select(condition, 0, cotangent)` for the `on_false` input.
#[derive(Clone)]
pub struct LinearSelectOperation<F> {
    /// Captured Boolean condition that drives the selection.
    condition: F,

    /// [`PhantomData`] marker tying the captured condition to the [`DataType`] it is interpreted against. The
    /// `fn() -> DataType` form indexes by [`DataType`] without owning one, so this operation's `Send` and `Sync`
    /// depend only on `F`.
    marker: PhantomData<fn() -> DataType>,
}

impl<F> LinearSelectOperation<F> {
    /// Creates a new [`LinearSelectOperation`] capturing the provided Boolean condition.
    #[inline]
    pub fn new(condition: F) -> Self {
        Self { condition, marker: PhantomData }
    }

    /// Returns the captured Boolean condition that drives the selection.
    #[inline]
    pub fn condition(&self) -> &F {
        &self.condition
    }
}

impl<F: Debug> Debug for LinearSelectOperation<F> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LinearSelectOperation").field("condition", &self.condition).finish()
    }
}

impl<F: Clone + Display> Display for LinearSelectOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Clone + Display> Operation<DataType> for LinearSelectOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[DataType],
        _region_interfaces: &[RegionInterface<DataType>],
    ) -> Result<Vec<DataType>, TypeError> {
        // The captured-condition select is linear in its two branch tangents, which share a type that is the output
        // type. The Boolean primal condition is captured as a factor (the primal trace already typed it), so it is not
        // revalidated here; only the branch tangents are checked.
        check_count!("input", input_types, 2, TypeError);
        if input_types[0] != input_types[1] {
            return Err(TypeError {
                message: format!(
                    "'select' on_true data type {} differs from on_false data type {}",
                    input_types[0], input_types[1],
                ),
            });
        }
        Ok(vec![input_types[0]])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("condition", &self.condition))
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for LinearSelectOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        SELECT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        // The two branch tangents are the operation's inputs; the captured Boolean condition is prepended as the
        // selection operand so that the underlying `select` shape inference (which expects the condition first) runs.
        check_count!("input", input_types, 2, TypeError);
        SelectOperation.infer_output_types(
            &[self.condition.r#type().into_owned(), input_types[0].clone(), input_types[1].clone()],
            &[],
        )
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SELECT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("condition", &self.condition))
    }
}

/// Interpretation materializes the captured condition factor into the interpreting value type through
/// [`CustomVjpResidual`](crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual) (an identity for
/// conditions captured as plain runtime values) and selects through the value-level [`Select`] capability, with the
/// materialized condition's [`SelectCondition`] view providing the condition representation of the active value
/// semantics (a decoded in-band [`bool`] for eager scalars, the value itself for arrays and staged tracers).
impl<F, C: Domain> InterpretableOperation<C> for LinearSelectOperation<F>
where
    C::Value: SelectCondition + Select<Condition = <C::Value as SelectCondition>::Condition>,
    F: crate::tracing_v2::operations::custom_derivatives::CustomVjpResidual<C::Value>,
    Self: Operation<C::Type>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![C::Value::select(&self.condition().residual_value()?.select_condition()?, &inputs[0], &inputs[1])?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate) for a [`LinearSelectOperation`].
impl<F: Value<Type = ArrayType>, C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C>
    for LinearSelectOperation<F>
where
    C::Operation: From<LinearSelectOperation<F>>,
{
}

/// Transpose rule for the captured-condition select. The forward linear map
/// `(t, f) ↦ select(condition, t, f)` routes the output cotangent into the branch the captured condition selected:
/// the `on_true` cotangent is `select(condition, cotangent, 0)` and the `on_false` cotangent is
/// `select(condition, 0, cotangent)`. The transposed select reuses the same captured condition, reconstructed from
/// `self` and staged back into the transpose builder. The impl is generic over the primary type `V::Type` and applies
/// wherever `LinearSelectOperation<F>` implements [`Operation`] for it (i.e., [`DataType`] and [`ArrayType`]).
impl<V: Value, O: Operation<V::Type>, F: Clone> TransposableOperation<V, O> for LinearSelectOperation<F>
where
    V::Type: DifferentiableType,
    Self: Operation<V::Type>,
    O: From<ZeroOperation<V::Type>> + From<LinearSelectOperation<F>>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(vec![
                MaybeZero::Zero(inputs[0].r#type().cotangent()),
                MaybeZero::Zero(inputs[1].r#type().cotangent()),
            ]),
            MaybeZero::Value(cotangent) => {
                let zero = MaybeZero::Zero(cotangent.r#type().into_owned()).materialize(context)?;
                let operation = || O::from(LinearSelectOperation::new(self.condition().clone()));
                let on_true = context.stage_operation(operation(), Vec::new(), &[cotangent.clone(), zero.clone()])?;
                check_count!("output", on_true, 1, ProgramError);
                let on_false = context.stage_operation(operation(), Vec::new(), &[zero, cotangent.clone()])?;
                check_count!("output", on_false, 1, ProgramError);
                let on_true_type = inputs[0].r#type().cotangent();
                let on_false_type = inputs[1].r#type().cotangent();
                Ok(vec![
                    MaybeZero::Value(on_true.into_iter().next().unwrap().unalign_cotangent(&on_true_type)?),
                    MaybeZero::Value(on_false.into_iter().next().unwrap().unalign_cotangent(&on_false_type)?),
                ])
            }
        }
    }
}

/// Partition-aware transpose rule for the primal [`SelectOperation`]. The Boolean condition (operand 0) has no
/// tangent space, so in a valid pushforward it is the known operand and the two branches (operands 1 and 2) are the
/// linear ones. The forward map `(on_true, on_false) ↦ select(condition, on_true, on_false)` routes the output
/// cotangent into the branch the known condition selected: the `on_true` cotangent is `select(condition, cotangent, 0)`
/// and the `on_false` cotangent is `select(condition, 0, cotangent)`. This reproduces the captured-condition
/// [`LinearSelectOperation`] transpose rule, reading the condition from the pullback through `operand_values` and
/// staging a primal `select` instead of folding the condition into a captured factor. The condition receives a
/// structural zero, and a zero output cotangent stays a structural zero.
///
/// The rule is generic over the primary type `V::Type` because it only reaches the branch type (`input_types[1]`), the
/// known condition operand value, and the primal `select`; it carries no rank- or shape-specific logic. It therefore
/// applies to both the array [`ArrayOperation::Select`](crate::backends::arrays::ArrayOperation) and the scalar
/// [`ScalarOperation::Select`](crate::backends::scalars::ScalarOperation) enum dispatch.
impl<V: Value, O> TransposableOperation<V, O> for SelectOperation
where
    V::Type: DifferentiableType,
    SelectOperation: Operation<V::Type>,
    O: Operation<V::Type> + From<ZeroOperation<V::Type>> + From<SelectOperation>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<V::Type>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
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
                // The condition is the known operand; the dispatch guarantees a `Known` operand carries its pullback
                // value, so read the tracer directly.
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
}

/// Forward-mode rule for [`SelectOperation`]: the primal output is `select(condition, on_true, on_false)` over
/// the input primals, and the tangent selects the branch tangents under the *same* primal condition (a `select` is
/// piecewise linear in its branches), with the condition carried as an ordinary primal operand edge. When both branch
/// tangents are canonical staged zeros, the output tangent is a canonical staged zero of the output type.
impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for SelectOperation
where
    C::Type: DifferentiableType,
    C::Operation: From<SelectOperation>,
    C::Value: ElementwiseDerivativeAlignment<C::Type>,
    SelectOperation: Operation<C::Type>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        let condition = &inputs[0];
        let on_true = &inputs[1];
        let on_false = &inputs[2];
        // Bind the primal and tangent selects through the context rather than the value-level `Select` capability:
        // binding works uniformly under staging and eager contexts, whereas eager value types select over their own
        // condition representations (for example, `Scalar` selects over `bool`).
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
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::EagerContext;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::constants::ZeroLike;
    use crate::operations::control_flow::Select;
    use crate::operations::math::Add;
    use crate::programs::types::Typed;
    use crate::tracing_v2::{DenseDifferentiate, jacrev};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::LinearSelectOperation;

    /// `f(x) = select(x > 0, 2x, 3x)` expressed over staged or differentiation-dual values of any context with
    /// [`Array`] semantics.
    fn piecewise_select<V>(x: V) -> V
    where
        V: crate::programs::Value<Type = crate::types::ArrayType>,
        V::DispatchDomain: crate::contexts::Context<
                Type = crate::types::ArrayType,
                Constant = Array,
                Operation = crate::backends::arrays::ArrayOperation<Array>,
            >,
    {
        let mask = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan).unwrap();
        let doubled = x.add(&x).unwrap();
        let tripled = doubled.add(&x).unwrap();
        Select::select(&mask, &doubled, &tripled).unwrap()
    }

    #[test]
    fn test_select_jacrev_computes_piecewise_derivative() {
        // Reverse mode through `f(x) = select(x > 0, 2x, 3x)` exercises the captured-condition select transpose:
        // the on_true cotangent is `select(condition, cotangent, 0)` and the on_false cotangent is
        // `select(condition, 0, cotangent)`.
        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacrev(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().values()[0], 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_linear_select_debug_omits_marker() {
        let operation = LinearSelectOperation::new(true);
        assert_eq!(format!("{operation:?}"), "LinearSelectOperation { condition: true }");
    }

    #[test]
    fn test_select_jacfwd_computes_piecewise_derivative() {
        // Forward mode through the same function exercises the captured-condition select under batched basis
        // tangents (the direct batched JVP path).
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
    fn test_scalar_select_jvp_and_gradient_flow_to_selected_branch() {
        // Differentiating `f(x, y) = select(x > y, 2x, 3y)` over `EagerContext<Scalar, ScalarOperation<Scalar>>` exercises the scalar select rule:
        // forward mode routes each branch tangent through the selected branch, and reverse mode routes the cotangent
        // there, so the derivative reaches only the selected branch's input.
        use crate::backends::scalars::{Scalar, ScalarOperation};
        use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};

        fn piecewise<V>(x: V, y: V) -> Result<V, crate::programs::ProgramError>
        where
            V: Clone + Compare<Output = V> + Select<Condition = V> + std::ops::Add<Output = V>,
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
    fn test_select_partitioned_transpose_matches_captured_condition_select_adjoint() {
        use crate::operations::BooleanLike;
        use crate::operations::control_flow::SelectOperation;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Condition `[true, false]` (known), branches and cotangent are length-two f64 vectors (linear branches).
        let condition = Array::vector(vec![1.0, 0.0]).as_boolean();
        // The branches are linear operands, so only their type enters the transpose; their values are unused.
        let on_true = Array::vector(vec![10.0, 20.0]);
        let cotangent = Array::vector(vec![5.0, 7.0]);
        let condition_type = <Array as crate::programs::types::Typed>::r#type(&condition).into_owned();
        let branch_type = <Array as crate::programs::types::Typed>::r#type(&on_true).into_owned();

        // Build `select(condition, on_true, on_false)` over the test enum, treat only the branches as linear, and
        // interpret the pullback on `[cotangent, condition]`.
        let mut builder = ProgramBuilder::<Array, crate::backends::arrays::ArrayOperation<Array>>::new();
        let condition_input = builder.add_input(condition_type.clone());
        let on_true_input = builder.add_input(branch_type.clone());
        let on_false_input = builder.add_input(branch_type.clone());
        let output = builder
            .add_instruction(SelectOperation, Vec::new(), vec![condition_input, on_true_input, on_false_input])
            .unwrap()[0];
        let program = builder
            .build::<(Array, Array, Array), Array>(vec![output], (Placeholder, Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1, 2]).unwrap();
        assert_eq!(pullback.output_ids().len(), 2, "the known condition input must receive no cotangent output");
        let branch_cotangents = pullback.interpret(vec![cotangent, condition]).unwrap();

        // The select adjoint routes the cotangent into each selected branch: under condition `[true, false]` the
        // `on_true` cotangent keeps the cotangent at the true batch items and zeroes the rest (`[5, 0]`), and the
        // `on_false` cotangent does the opposite (`[0, 7]`).
        assert_eq!(branch_cotangents.len(), 2);
        assert_eq!(branch_cotangents[0].to_f64s(), vec![5.0, 0.0]);
        assert_eq!(branch_cotangents[1].to_f64s(), vec![0.0, 7.0]);
    }
}
